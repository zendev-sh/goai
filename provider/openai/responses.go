package openai

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// serverExecutedItemTypes lists Responses API output item types that the
// provider runs server-side (no client execution). When these items appear in
// the response output, their full payload must round-trip on the assistant
// turn so the model retains context across follow-up requests.
var serverExecutedItemTypes = map[string]bool{
	"web_search_call":       true,
	"file_search_call":      true,
	"code_interpreter_call": true,
	"image_generation_call": true,
	"local_shell_call":      true,
	"mcp_call":              true,
	"mcp_list_tools":        true,
	"mcp_approval_request":  true,
	"computer_call":         true,
}

func isServerExecutedItem(t string) bool { return serverExecutedItemTypes[t] }

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
	if auto, ok := params.ProviderOptions["goaiAutoPreviousResponseID"].(bool); ok && auto {
		body["input"] = convertAutoContinuationInput(params.Messages)
	}

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
		"structuredOutputs":          true,
		"strictJsonSchema":           true,
		"useResponsesAPI":            true,
		"goaiAutoPreviousResponseID": true,
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
			var message map[string]any
			var messageContent []map[string]any
			var messageID, messagePhase string
			appendText := func(part provider.Part) {
				id, _ := part.ProviderOptions["itemId"].(string)
				phase, _ := part.ProviderOptions["phase"].(string)
				if part.Text == "" && phase == "" && id == "" {
					return
				}
				if id != messageID || phase != messagePhase {
					message = nil
				}
				if message == nil {
					messageContent = nil
					messageID, messagePhase = id, phase
					message = map[string]any{
						"type":    "message",
						"role":    "assistant",
						"content": messageContent,
					}
					if phase != "" {
						message["phase"] = phase
					}
					items = append(items, message)
				}
				content := map[string]any{
					"type":        "output_text",
					"text":        part.Text,
					"annotations": []any{},
				}
				if refusal, _ := part.ProviderOptions["refusal"].(bool); refusal {
					content = map[string]any{"type": "refusal", "refusal": part.Text}
				}
				messageContent = append(messageContent, content)
				message["content"] = messageContent
			}

			for _, part := range msg.Content {
				switch part.Type {
				case provider.PartText:
					appendText(part)
				case provider.PartReasoning:
					if item, ok := reasoningInputItem(part); ok {
						message = nil
						items = append(items, item)
					} else if part.Text != "" {
						appendText(provider.Part{Text: part.Text})
					}
				case provider.PartToolCall:
					message = nil
					// Server-executed tool items (web_search_call,
					// file_search_call, ...) round-trip verbatim so the model
					// sees the same context across turns.
					if raw, ok := part.ProviderOptions["rawItem"].(map[string]any); ok {
						items = append(items, raw)
						break
					}
					items = append(items, map[string]any{
						"type":      "function_call",
						"call_id":   part.ToolCallID,
						"name":      part.ToolName,
						"arguments": string(part.ToolInput),
					})
				}
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
						"type": "input_file",
					}
					if part.RemoteRef != nil {
						item["file_id"] = part.RemoteRef.ID
					} else {
						item["file_data"] = part.URL
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

func convertAutoContinuationInput(msgs []provider.Message) []map[string]any {
	var toolMessages []provider.Message
	for i := len(msgs) - 1; i >= 0 && msgs[i].Role == provider.RoleTool; i-- {
		toolMessages = append(toolMessages, msgs[i])
	}
	slices.Reverse(toolMessages)
	return convertToResponsesInput(toolMessages)
}

func reasoningInputItem(part provider.Part) (map[string]any, bool) {
	openAI, ok := part.ProviderOptions["openai"].(map[string]any)
	if !ok {
		return nil, false
	}
	itemID, _ := openAI["itemId"].(string)
	encryptedContent, _ := openAI["encryptedContent"].(string)
	if itemID == "" {
		return nil, false
	}
	item := map[string]any{"type": "reasoning"}
	if itemID != "" {
		item["id"] = itemID
	}
	if encryptedContent != "" {
		item["encrypted_content"] = encryptedContent
	}
	item["summary"] = []map[string]any{}
	if part.Text != "" {
		item["summary"] = []map[string]any{{"type": "summary_text", "text": part.Text}}
	}
	return item, true
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
	// lastSummary is the summary_index of the previous delta, or -1 before the
	// first one. Segments are only delimited by that index, never by the text.
	lastSummary int
}

// Bound the completed output retained for client-side replay across a stream.
var maxResponsesReplayBytes = 64 << 20

func activeReasoningForEvent(active map[int]*responsesReasoning, current int, itemID string) (int, *responsesReasoning) {
	for index, reasoning := range active {
		if reasoning.canonicalID == itemID {
			return index, reasoning
		}
	}
	if reasoning, ok := active[current]; ok {
		return current, reasoning
	}
	return -1, nil
}

func openAIReasoningPart(itemID, text, encryptedContent string) provider.Part {
	openAI := map[string]any{"itemId": itemID}
	if encryptedContent != "" {
		openAI["encryptedContent"] = encryptedContent
	}
	return provider.Part{
		Type:            provider.PartReasoning,
		Text:            text,
		ProviderOptions: map[string]any{"openai": openAI},
	}
}

// summarySeparator delimits consecutive reasoning summaries. Their boundary
// exists only in summary_index, so concatenating the deltas runs two summaries
// together and merges the markdown at the seam.
const summarySeparator = "\n\n"

// streamResponses parses SSE from the OpenAI Responses API with the default
// event-level idle timeout.
func streamResponses(ctx context.Context, body io.ReadCloser, out chan<- provider.StreamChunk) {
	streamResponsesWithConfig(ctx, body, out, defaultResponsesStreamConfig())
}

// responsesReplayContent assembles the authoritative terminal output, or the
// completed item events when a gateway omits that output (including [DONE]).
func responsesReplayContent(items []json.RawMessage, completed map[int]json.RawMessage, phases map[string]string) ([]provider.Part, error) {
	if len(items) == 0 {
		for _, index := range slices.Sorted(maps.Keys(completed)) {
			items = append(items, completed[index])
		}
	}
	if len(items) == 0 {
		return nil, nil
	}
	payload, err := json.Marshal(map[string]any{"output": items})
	if err != nil {
		return nil, err
	}
	parsed, err := parseResponsesResult(payload)
	if err != nil {
		return nil, err
	}
	for i := range parsed.Content {
		part := &parsed.Content[i]
		if part.Type == provider.PartText && part.ProviderOptions["phase"] == nil {
			id, _ := part.ProviderOptions["itemId"].(string)
			if phase := phases[id]; phase != "" {
				part.ProviderOptions["phase"] = phase
			}
		}
	}
	return parsed.Content, nil
}

func streamResponsesWithConfig(
	ctx context.Context,
	body io.ReadCloser,
	out chan<- provider.StreamChunk,
	config responsesStreamConfig,
) {
	defer close(out)

	reader := newResponsesEventReader(ctx, body)
	defer reader.close()

	idleTimer := newResponsesIdleTimer(config.idleTimeout)
	defer idleTimer.stop()

	var usage provider.Usage
	var hasFunctionCall bool

	activeTools := make(map[int]*responsesToolCall)
	activeReasoning := make(map[int]*responsesReasoning)
	messageOptions := make(map[int]map[string]any)
	messagePhases := make(map[string]string)
	outputItems := make(map[int]json.RawMessage)
	outputItemsBytes := 0
	currentReasoningIdx := -1
	finish := func(items []json.RawMessage, response provider.ResponseMetadata, reason provider.FinishReason) {
		content, err := responsesReplayContent(items, outputItems, messagePhases)
		if err != nil {
			trySendResponsesError(ctx, out, err)
			return
		}
		// Both supported terminal forms must flush complete buffered tool args.
		for _, index := range slices.Sorted(maps.Keys(activeTools)) {
			active := activeTools[index]
			if remaining := active.args.String(); remaining != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: active.id, ToolName: active.name, ToolInput: remaining}) {
					return
				}
			}
		}
		if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkStepFinish, FinishReason: reason}) {
			return
		}
		provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, Content: content, Usage: usage, Response: response})
	}

	for {
		var read responsesReadResult
		select {
		case <-ctx.Done():
			trySendResponsesError(ctx, out, ctx.Err())
			return
		case <-idleTimer.c:
			if err := ctx.Err(); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			err := &StreamIdleTimeoutError{
				Provider: responsesStreamProvider,
				API:      responsesStreamAPI,
				Idle:     config.idleTimeout,
			}
			trySendResponsesError(ctx, out, err)
			return
		case read = <-reader.results:
		}

		if err := ctx.Err(); err != nil {
			trySendResponsesError(ctx, out, err)
			return
		}
		if read.err != nil {
			trySendResponsesError(ctx, out, newStreamProtocolError("", "stream read failed", read.err))
			return
		}
		if read.eof {
			trySendResponsesError(ctx, out, newStreamProtocolError("", "stream ended before a terminal event", nil))
			return
		}

		data := string(read.event.Data)

		if data == "[DONE]" {
			if config.allowDone {
				finish(nil, provider.ResponseMetadata{}, mapResponsesFinishReason("response.completed", "", hasFunctionCall))
				return
			}
			trySendResponsesError(ctx, out, newStreamProtocolError(read.event.Type, "stream ended with [DONE] before a terminal event", nil))
			return
		}
		eventType, err := responsesEventType(read.event)
		if err != nil {
			trySendResponsesError(ctx, out, err)
			return
		}

		switch eventType {
		case "response.output_text.delta":
			var ev struct {
				Delta       *string `json:"delta"`
				OutputIndex int     `json:"output_index"`
				ItemID      string  `json:"item_id"`
			}
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.Delta == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "delta"))
				return
			}
			if *ev.Delta != "" {
				metadata := maps.Clone(messageOptions[ev.OutputIndex])
				if metadata == nil && ev.ItemID != "" {
					metadata = map[string]any{"itemId": ev.ItemID}
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: *ev.Delta, Metadata: metadata}) {
					return
				}
			}

		case "response.refusal.delta":
			var ev struct {
				Delta *string `json:"delta"`
			}
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.Delta == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "delta"))
				return
			}
			if *ev.Delta != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: *ev.Delta}) {
					return
				}
			}

		case "response.reasoning_summary_text.delta":
			var ev struct {
				ItemID       *string `json:"item_id"`
				SummaryIndex *int    `json:"summary_index"`
				Delta        *string `json:"delta"`
			}
			// Preseeding keeps omitted legacy fields at their zero values while
			// still letting json.Unmarshal expose an explicit null as nil.
			ev.ItemID = new(string)
			ev.SummaryIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.ItemID == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "item_id"))
				return
			}
			if ev.SummaryIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "summary_index"))
				return
			}
			if ev.Delta == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "delta"))
				return
			}
			if *ev.Delta != "" {
				// Use canonical ID from activeReasoning if available.
				id, text := *ev.ItemID, *ev.Delta
				if idx, ar := activeReasoningForEvent(activeReasoning, currentReasoningIdx, *ev.ItemID); idx >= 0 {
					id = ar.canonicalID
					// A new summary_index opens a separate summary: carry the
					// boundary in the text, since the deltas never bring it.
					if ar.lastSummary >= 0 && *ev.SummaryIndex != ar.lastSummary {
						text = summarySeparator + text
					}
					ar.lastSummary = *ev.SummaryIndex
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkReasoning,
					Text: text,
					Metadata: map[string]any{
						"reasoningId": fmt.Sprintf("%s:%d", id, *ev.SummaryIndex),
					},
				}) {
					return
				}
			}

		case "response.reasoning_text.delta":
			// Raw reasoning text, as opposed to a summary of it, filling the
			// reasoning item's reasoning_text content parts. DeepSeek's
			// Responses API streams its whole chain of thought this way.
			var ev struct {
				ItemID       *string `json:"item_id"`
				ContentIndex *int    `json:"content_index"`
				Delta        *string `json:"delta"`
			}
			ev.ItemID = new(string)
			ev.ContentIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.ItemID == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "item_id"))
				return
			}
			if ev.ContentIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "content_index"))
				return
			}
			if ev.Delta == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "delta"))
				return
			}
			if *ev.Delta != "" {
				id := *ev.ItemID
				if idx, ar := activeReasoningForEvent(activeReasoning, currentReasoningIdx, *ev.ItemID); idx >= 0 {
					id = ar.canonicalID
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkReasoning,
					Text: *ev.Delta,
					Metadata: map[string]any{
						"reasoningId": fmt.Sprintf("%s:%d", id, *ev.ContentIndex),
					},
				}) {
					return
				}
			}

		case "response.reasoning_summary_part.added":
			var ev struct {
				ItemID         *string `json:"item_id"`
				OutputIndex    *int    `json:"output_index"`
				SummaryIndex   *int    `json:"summary_index"`
				SequenceNumber *int    `json:"sequence_number"`
				Part           *struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"part"`
			}
			ev.ItemID = new(string)
			ev.OutputIndex = new(int)
			ev.SummaryIndex = new(int)
			ev.SequenceNumber = new(int)
			ev.Part = &struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{}
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.ItemID == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "item_id"))
				return
			}
			if ev.OutputIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "output_index"))
				return
			}
			if ev.SummaryIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "summary_index"))
				return
			}
			if ev.SequenceNumber == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "sequence_number"))
				return
			}
			if ev.Part == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "part"))
				return
			}
			// New summary segments do not emit chunks, but decoding validates the
			// recognized event before it counts as stream activity.

		case "response.output_item.added":
			var ev struct {
				OutputIndex *int `json:"output_index"`
				Item        *struct {
					Type   *string `json:"type"`
					ID     string  `json:"id"`
					CallID string  `json:"call_id"`
					Name   string  `json:"name"`
					Phase  string  `json:"phase"`
				} `json:"item"`
			}
			ev.OutputIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.OutputIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "output_index"))
				return
			}
			if ev.Item == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "item"))
				return
			}
			if ev.Item.Type == nil || *ev.Item.Type == "" {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "item.type"))
				return
			}
			switch *ev.Item.Type {
			case "message":
				messageOptions[*ev.OutputIndex] = map[string]any{"itemId": ev.Item.ID}
				if ev.Item.Phase != "" {
					messageOptions[*ev.OutputIndex]["phase"] = ev.Item.Phase
					messagePhases[ev.Item.ID] = ev.Item.Phase
				}
			case "function_call":
				hasFunctionCall = true
				activeTools[*ev.OutputIndex] = &responsesToolCall{
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
				activeReasoning[*ev.OutputIndex] = &responsesReasoning{
					canonicalID: ev.Item.ID,
					lastSummary: -1,
				}
				currentReasoningIdx = *ev.OutputIndex
			}

		case "response.function_call_arguments.delta":
			var ev struct {
				OutputIndex *int    `json:"output_index"`
				Delta       *string `json:"delta"`
			}
			ev.OutputIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.OutputIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "output_index"))
				return
			}
			if ev.Delta == nil {
				trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "delta"))
				return
			}
			if *ev.Delta != "" {
				active := activeTools[*ev.OutputIndex]
				if active == nil {
					active = &responsesToolCall{}
					activeTools[*ev.OutputIndex] = active
				}
				active.args.WriteString(*ev.Delta)

				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCallDelta,
					ToolCallID: active.id,
					ToolName:   active.name,
					ToolInput:  *ev.Delta,
				}) {
					return
				}

				// The call is resolved on function_call_arguments.done. Finalizing
				// as soon as the buffer parses assumes no further delta arrives,
				// and any that does becomes a second, bogus call.
			}

		case "response.function_call_arguments.done":
			var ev struct {
				OutputIndex *int `json:"output_index"`
			}
			ev.OutputIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.OutputIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "output_index"))
				return
			}
			if active := activeTools[*ev.OutputIndex]; active != nil {
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

		case "response.output_item.done":
			var ev struct {
				OutputIndex *int            `json:"output_index"`
				Item        json.RawMessage `json:"item"`
			}
			ev.OutputIndex = new(int)
			if err := decodeResponsesEvent(eventType, read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, err)
				return
			}
			if ev.OutputIndex == nil {
				trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "output_index"))
				return
			}
			var itemHead *struct {
				Type *string `json:"type"`
			}
			if len(ev.Item) > 0 {
				if err := decodeResponsesEvent(eventType, ev.Item, &itemHead); err != nil {
					trySendResponsesError(ctx, out, err)
					return
				}
				if itemHead == nil {
					trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "event payload has null item", nil))
					return
				}
				if itemHead.Type == nil || *itemHead.Type == "" {
					trySendResponsesError(ctx, out, missingResponsesEventField(eventType, "item.type"))
					return
				}
			}
			itemType := ""
			if itemHead != nil {
				itemType = *itemHead.Type
				outputItemsBytes += len(ev.Item) - len(outputItems[*ev.OutputIndex])
				if outputItemsBytes > maxResponsesReplayBytes {
					trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "replay output exceeds size limit", nil))
					return
				}
				outputItems[*ev.OutputIndex] = ev.Item
			}
			switch itemType {
			case "reasoning":
				var item struct {
					ID               string `json:"id"`
					EncryptedContent string `json:"encrypted_content"`
				}
				if err := decodeResponsesEvent(eventType, ev.Item, &item); err != nil {
					trySendResponsesError(ctx, out, err)
					return
				}
				if active := activeReasoning[*ev.OutputIndex]; active != nil && item.EncryptedContent != "" {
					id := active.canonicalID
					if id == "" {
						id = item.ID
					}
					index := active.lastSummary
					if index < 0 {
						index = 0
					}
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type: provider.ChunkReasoning,
						Metadata: map[string]any{
							"reasoningId": fmt.Sprintf("%s:%d", id, index),
							"openai": map[string]any{
								"itemId":           id,
								"encryptedContent": item.EncryptedContent,
							},
						},
					}) {
						return
					}
				}
				delete(activeReasoning, *ev.OutputIndex)
				if currentReasoningIdx == *ev.OutputIndex {
					currentReasoningIdx = -1
				}
			default:
				if isServerExecutedItem(itemType) {
					// Capture the full server-executed item payload and
					// emit a ChunkToolCall so it round-trips into the
					// assistant turn via ToolCall.Metadata.
					var itemIdentity struct {
						ID   *string `json:"id"`
						Name *string `json:"name"`
					}
					itemIdentity.ID = new(string)
					itemIdentity.Name = new(string)
					if err := decodeResponsesEvent(eventType, ev.Item, &itemIdentity); err != nil {
						trySendResponsesError(ctx, out, err)
						return
					}
					if itemIdentity.ID == nil {
						trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "item.id"))
						return
					}
					if itemIdentity.Name == nil {
						trySendResponsesError(ctx, out, nullResponsesEventField(eventType, "item.name"))
						return
					}
					var rawItem map[string]any
					if err := decodeResponsesEvent(eventType, ev.Item, &rawItem); err != nil {
						trySendResponsesError(ctx, out, err)
						return
					}
					id := *itemIdentity.ID
					name := *itemIdentity.Name
					if name == "" {
						name = itemType
					}
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCall,
						ToolCallID: id,
						ToolName:   name,
						Metadata: map[string]any{
							"providerExecuted": true,
							"rawItem":          rawItem,
						},
					}) {
						return
					}
				} else if active := activeTools[*ev.OutputIndex]; active != nil {
					// Safety net for an item closed without its arguments.done.
					// The normal path resets the buffer, so this stays empty.
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
				delete(activeTools, *ev.OutputIndex)
			}

		case "response.completed", "response.incomplete":
			var ev struct {
				Response *struct {
					ID                string            `json:"id"`
					Model             string            `json:"model"`
					Output            []json.RawMessage `json:"output"`
					IncompleteDetails *struct {
						Reason string `json:"reason"`
					} `json:"incomplete_details"`
					Usage *struct {
						InputTokens         *int `json:"input_tokens"`
						OutputTokens        *int `json:"output_tokens"`
						OutputTokensDetails *struct {
							ReasoningTokens int `json:"reasoning_tokens"`
						} `json:"output_tokens_details"`
						InputTokensDetails *struct {
							CachedTokens int `json:"cached_tokens"`
						} `json:"input_tokens_details"`
					} `json:"usage"`
				} `json:"response"`
			}
			if err := json.Unmarshal(read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "malformed terminal event", err))
				return
			}
			if ev.Response == nil {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "terminal event is missing response", nil))
				return
			}
			if responseUsage := ev.Response.Usage; responseUsage != nil {
				if responseUsage.InputTokens == nil || responseUsage.OutputTokens == nil {
					trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "terminal event has incomplete response usage", nil))
					return
				}
				usage.InputTokens = *responseUsage.InputTokens
				usage.OutputTokens = *responseUsage.OutputTokens
				usage.TotalTokens = *responseUsage.InputTokens + *responseUsage.OutputTokens
				if responseUsage.OutputTokensDetails != nil {
					usage.ReasoningTokens = responseUsage.OutputTokensDetails.ReasoningTokens
				}
				if responseUsage.InputTokensDetails != nil {
					usage.CacheReadTokens = responseUsage.InputTokensDetails.CachedTokens
				}
				usage.InputTokens -= usage.CacheReadTokens
			}

			var incompleteReason string
			if ev.Response.IncompleteDetails != nil {
				incompleteReason = ev.Response.IncompleteDetails.Reason
			}
			finishReason := mapResponsesFinishReason(eventType, incompleteReason, hasFunctionCall)
			finish(ev.Response.Output, provider.ResponseMetadata{ID: ev.Response.ID, Model: ev.Response.Model}, finishReason)
			return

		case "response.failed":
			var ev struct {
				Response *struct {
					Error struct {
						Message string `json:"message"`
						Code    string `json:"code"`
					} `json:"error"`
				} `json:"response"`
			}
			if err := json.Unmarshal(read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "malformed terminal event", err))
				return
			}
			if ev.Response == nil {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "terminal event is missing response", nil))
				return
			}
			if !provider.TrySend(ctx, out, responsesStreamError(data, ev.Response.Error.Message, ev.Response.Error.Code, "response failed")) {
				return
			}
			return

		case "error":
			// OpenAI documents flat message/code fields, but production often nests them under error.
			var ev struct {
				Message string `json:"message"`
				Code    string `json:"code"`
				Error   *struct {
					Message string `json:"message"`
					Code    string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(read.event.Data, &ev); err != nil {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "malformed terminal event", err))
				return
			}
			// Prefer the nested error fields when present, but fall back to
			// the flat fields per-field so a partial nested object (e.g. a
			// nested code with no nested message) does not clobber a flat
			// message/code with an empty string.
			msg, code := ev.Message, ev.Code
			if ev.Error != nil {
				msg = cmp.Or(ev.Error.Message, msg)
				code = cmp.Or(ev.Error.Code, code)
			}
			if msg == "" && code == "" {
				trySendResponsesError(ctx, out, newStreamProtocolError(eventType, "terminal error event is missing error details", nil))
				return
			}
			if !provider.TrySend(ctx, out, responsesStreamError(data, msg, code, "stream error")) {
				return
			}
			return
		}

		// Projection may block on downstream backpressure. Restart the provider
		// idle window only after projection completes so consumer latency cannot
		// be misreported as provider inactivity.
		idleTimer.reset(config.idleTimeout)
	}
}

func responsesStreamError(data, msg, code, defaultMsg string) provider.StreamChunk {
	msg = cmp.Or(msg, defaultMsg)
	switch code {
	case "context_length_exceeded", "context_overflow", "max_tokens":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.ContextOverflowError{Message: msg, ResponseBody: data}}
	case "insufficient_quota":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: "Quota exceeded. Check your plan and billing details.", IsRetryable: false}}
	case "usage_not_included":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: "Usage not included in response. Check your plan supports usage reporting for this model.", IsRetryable: false}}
	case "invalid_prompt":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg, IsRetryable: false}}
	case "rate_limit_exceeded", "429":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg, StatusCode: 429, IsRetryable: true}}
	case "server_error", "503", "502", "500":
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg, IsRetryable: true}}
	default:
		return provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg}}
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
		Phase   string `json:"phase"`
		Role    string `json:"role"`
		Content []struct {
			Type        string                `json:"type"`
			Text        string                `json:"text"`
			Refusal     string                `json:"refusal"`
			Annotations []responsesAnnotation `json:"annotations,omitempty"`
			Logprobs    *json.RawMessage      `json:"logprobs,omitempty"`
		} `json:"content,omitempty"`

		// function_call fields
		CallID    string `json:"call_id,omitempty"`
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`

		// reasoning fields
		ID               string `json:"id,omitempty"`
		EncryptedContent string `json:"encrypted_content,omitempty"`
		Summary          []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"summary,omitempty"`
	} `json:"output"`

	Usage *struct {
		InputTokens         int `json:"input_tokens"`
		OutputTokens        int `json:"output_tokens"`
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

	// Side-channel raw parse so server-executed items (web_search_call etc.)
	// can be round-tripped verbatim on the assistant turn -- the typed struct
	// only models the subset of fields we explicitly consume.
	var rawOutput struct {
		Output []json.RawMessage `json:"output"`
	}
	_ = json.Unmarshal(body, &rawOutput)

	result := &provider.GenerateResult{
		Response: provider.ResponseMetadata{
			ID:    resp.ID,
			Model: resp.Model,
		},
	}

	// Extract text, tool calls, sources, logprobs from output.
	var textParts []string
	var reasoningParts []string
	var reasoningItems []provider.Part
	var hasFunctionCall bool
	providerMeta := map[string]any{}
	var allLogprobs []any

	for i, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" || c.Type == "refusal" {
					opts := map[string]any{"itemId": item.ID}
					if item.Phase != "" {
						opts["phase"] = item.Phase
					}
					text := c.Text
					if c.Type == "refusal" {
						text = c.Refusal
						opts["refusal"] = true
					}
					result.Content = append(result.Content, provider.Part{Type: provider.PartText, Text: text, ProviderOptions: opts})
					if text != "" {
						textParts = append(textParts, text)
					}
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
			result.Content = append(result.Content, provider.Part{Type: provider.PartToolCall, ToolCallID: item.CallID, ToolName: item.Name, ToolInput: json.RawMessage(item.Arguments)})
			result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(item.Arguments),
			})
		case "reasoning":
			var itemText []string
			for _, s := range item.Summary {
				if s.Text != "" {
					reasoningParts = append(reasoningParts, s.Text)
					itemText = append(itemText, s.Text)
					reasoning, _ := providerMeta["reasoning"].([]map[string]any)
					providerMeta["reasoning"] = append(reasoning, map[string]any{
						"type": s.Type,
						"text": s.Text,
					})
				}
			}
			// Raw reasoning comes as reasoning_text content parts instead of
			// a summary, as DeepSeek's Responses API returns it.
			for _, c := range item.Content {
				if c.Type == "reasoning_text" && c.Text != "" {
					reasoningParts = append(reasoningParts, c.Text)
					itemText = append(itemText, c.Text)
					reasoning, _ := providerMeta["reasoning"].([]map[string]any)
					providerMeta["reasoning"] = append(reasoning, map[string]any{
						"type": c.Type,
						"text": c.Text,
					})
				}
			}
			if len(itemText) > 0 || item.EncryptedContent != "" {
				reasoningItems = append(reasoningItems, openAIReasoningPart(item.ID, strings.Join(itemText, summarySeparator), item.EncryptedContent))
				result.Content = append(result.Content, reasoningItems[len(reasoningItems)-1])
			}
		default:
			if isServerExecutedItem(item.Type) && i < len(rawOutput.Output) {
				var raw map[string]any
				if err := json.Unmarshal(rawOutput.Output[i], &raw); err == nil {
					id, _ := raw["id"].(string)
					name, _ := raw["name"].(string)
					if name == "" {
						name = item.Type
					}
					result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
						ID:   id,
						Name: name,
						Metadata: map[string]any{
							"providerExecuted": true,
							"rawItem":          raw,
						},
					})
					result.Content = append(result.Content, provider.Part{Type: provider.PartToolCall, ToolCallID: id, ToolName: name, ProviderOptions: map[string]any{"providerExecuted": true, "rawItem": raw}})
				}
			}
		}
	}

	result.Text = strings.Join(textParts, "")
	// Each entry is a distinct summary; joining them bare merges the markdown
	// at the seam, same as in the streaming path.
	result.Reasoning = strings.Join(reasoningParts, summarySeparator)
	result.ReasoningParts = reasoningItems

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
