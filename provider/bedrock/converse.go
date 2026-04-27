package bedrock

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"sync/atomic"

	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// bedrockDocumentCounter generates unique fallback document names.
var bedrockDocumentCounter atomic.Int64

// bedrockNameRe matches characters allowed by AWS Bedrock Converse API:
// alphanumeric, whitespace, hyphens, parentheses, square brackets.
var bedrockNameRe = regexp.MustCompile(`[^a-zA-Z0-9\s\-\(\)\[\]]`)

// bedrockMultiHyphen collapses consecutive hyphens from replaced characters.
var bedrockMultiHyphen = regexp.MustCompile(`-{2,}`)

// bedrockMultiSpace collapses consecutive whitespace (AWS rejects it).
var bedrockMultiSpace = regexp.MustCompile(`\s{2,}`)

// sanitizeDocumentName ensures the name conforms to AWS Bedrock requirements:
// only [a-zA-Z0-9\s\-\(\)\[\]], no consecutive whitespace, max 200 chars.
// Replaces invalid chars (dots, underscores, Unicode, etc.) with hyphens,
// keeping extension info (e.g., "report.pdf" → "report-pdf").
// Falls back to auto-generated "document-N" for empty/non-Latin filenames.
//
// Note: Vercel AI SDK strips extension entirely ("report.pdf" → "report") and
// does NOT sanitize other chars (underscores, slashes still pass through and
// may cause AWS validation errors). GoAI sanitizes all invalid chars.
func sanitizeDocumentName(name string) string {
	// Replace invalid characters with hyphens (dots, underscores, Unicode, etc.).
	name = bedrockNameRe.ReplaceAllString(name, "-")
	// Collapse consecutive hyphens.
	name = bedrockMultiHyphen.ReplaceAllString(name, "-")
	// Collapse consecutive whitespace.
	name = bedrockMultiSpace.ReplaceAllString(name, " ")
	name = strings.Trim(name, "- ")
	// Truncate to 200 chars.
	if len(name) > 200 {
		name = name[:200]
	}
	if name == "" {
		n := bedrockDocumentCounter.Add(1)
		name = fmt.Sprintf("document-%d", n)
	}
	return name
}

// buildConverseRequest builds a Bedrock Converse API request body from GenerateParams.
func buildConverseRequest(params provider.GenerateParams, modelID string) map[string]any {
	body := make(map[string]any)

	// System prompt.
	if params.System != "" {
		systemBlocks := []map[string]any{{"text": params.System}}
		// Add cache point if prompt caching is enabled.
		if params.PromptCaching {
			systemBlocks = append(systemBlocks, map[string]any{
				"cachePoint": map[string]any{"type": "default"},
			})
		}
		body["system"] = systemBlocks
	}

	// Messages.
	convMsgs := convertMessages(params.Messages)
	body["messages"] = convMsgs

	// Inference config.
	inferenceConfig := make(map[string]any)
	if params.MaxOutputTokens > 0 {
		inferenceConfig["maxTokens"] = params.MaxOutputTokens
	}
	if params.Temperature != nil {
		inferenceConfig["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		inferenceConfig["topP"] = *params.TopP
	}
	if len(params.StopSequences) > 0 {
		inferenceConfig["stopSequences"] = params.StopSequences
	}
	// TopK from provider options.
	if params.ProviderOptions != nil {
		if topK, ok := params.ProviderOptions["topK"]; ok {
			inferenceConfig["topK"] = topK
		}
	}
	if len(inferenceConfig) > 0 {
		body["inferenceConfig"] = inferenceConfig
	}

	// Per-request headers (extracted in transport layer before JSON marshaling).
	if len(params.Headers) > 0 {
		body["_headers"] = params.Headers
	}

	// For Anthropic models, request the stop_sequence field in additional response.
	if strings.Contains(modelID, "anthropic") {
		body["additionalModelResponseFieldPaths"] = []string{"/delta/stop_sequence"}
	}

	// Tool config.
	if len(params.Tools) > 0 {
		tools := make([]map[string]any, 0, len(params.Tools))
		for _, t := range params.Tools {
			schema := any(map[string]any{"type": "object"})
			if len(t.InputSchema) > 0 {
				if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
					schema = map[string]any{"type": "object"}
				}
			}
			// Bedrock Converse API requires non-empty description.
			// Provider-defined tools (computer-use) may omit it.
			desc := cmp.Or(t.Description, t.Name)
			tools = append(tools, map[string]any{
				"toolSpec": map[string]any{
					"name":        t.Name,
					"description": desc,
					"inputSchema": map[string]any{"json": schema},
				},
			})
		}
		toolConfig := map[string]any{"tools": tools}

		// Tool choice.
		switch params.ToolChoice {
		case "auto":
			toolConfig["toolChoice"] = map[string]any{"auto": map[string]any{}}
		case "any", "required":
			toolConfig["toolChoice"] = map[string]any{"any": map[string]any{}}
		case "none":
			// Don't send toolConfig at all if none.
			toolConfig = nil
		case "":
			// No tool choice specified; use default.
		default:
			// Specific tool name.
			toolConfig["toolChoice"] = map[string]any{
				"tool": map[string]any{"name": params.ToolChoice},
			}
		}
		if toolConfig != nil {
			body["toolConfig"] = toolConfig
		}
	}

	return body
}

// ensureToolConfigForHistory scans provider messages for toolUse/toolResult
// references and synthesizes a minimal toolConfig in the request body.
// This is required because Bedrock Converse API returns an error when messages
// contain toolUse or toolResult content blocks but toolConfig is absent.
//
// This scenario occurs after an Esc interrupt: the conversation history
// retains tool blocks from the aborted turn, and if the subsequent request
// has no tools (e.g., compaction agent, or a retry without tools), Bedrock
// rejects it. The fix creates placeholder tool definitions from the tool
// names found in the message history.
func ensureToolConfigForHistory(body map[string]any, msgs []provider.Message) {
	// Already has toolConfig - nothing to do.
	if _, ok := body["toolConfig"]; ok {
		return
	}

	// Scan messages for tool-call parts to extract tool names.
	toolNames := make(map[string]bool)
	for _, msg := range msgs {
		for _, p := range msg.Content {
			if p.Type == provider.PartToolCall && p.ToolName != "" {
				toolNames[p.ToolName] = true
			}
		}
	}

	if len(toolNames) == 0 {
		return
	}

	// Build minimal toolConfig with placeholder tool definitions.
	// Bedrock requires name + description + inputSchema for each tool.
	tools := make([]map[string]any, 0, len(toolNames))
	for name := range toolNames {
		tools = append(tools, map[string]any{
			"toolSpec": map[string]any{
				"name":        name,
				"description": name,
				"inputSchema": map[string]any{"json": map[string]any{"type": "object"}},
			},
		})
	}
	body["toolConfig"] = map[string]any{"tools": tools}
}

// convertMessages maps provider.Message slice to Bedrock Converse format.
// Tool-role messages are merged into user role per Bedrock convention.
func convertMessages(msgs []provider.Message) []map[string]any {
	// Pre-process: fix orphaned tool-calls, merge consecutive roles, reorder parts.
	msgs = provider.NormalizeToolMessages(msgs)
	msgs = provider.ReorderAssistantParts(msgs)

	var result []map[string]any
	// Track document names across the conversation. Bedrock requires
	// unique names. Append "-N" suffix for duplicates.
	docNames := make(map[string]int)

	for _, msg := range msgs {
		role := string(msg.Role)
		content := convertParts(msg.Content, docNames)

		if len(content) == 0 {
			continue
		}

		// Bedrock doesn't have a "tool" role -- tool results go under "user".
		if role == "tool" {
			role = "user"
		}

		// Bedrock requires alternating user/assistant. Merge consecutive same-role
		// (safety net - NormalizeToolMessages already merged at provider.Message level).
		if len(result) > 0 && result[len(result)-1]["role"] == role {
			existing, ok := result[len(result)-1]["content"].([]map[string]any)
			if ok {
				result[len(result)-1]["content"] = append(existing, content...)
				continue
			}
		}

		result = append(result, map[string]any{
			"role":    role,
			"content": content,
		})
	}

	return result
}

// convertParts maps provider.Part slice to Bedrock content blocks.
// docNames tracks document names across the conversation for uniqueness.
func convertParts(parts []provider.Part, docNames map[string]int) []map[string]any {
	var blocks []map[string]any

	for _, p := range parts {
		switch p.Type {
		case provider.PartText:
			if p.Text != "" {
				blocks = append(blocks, map[string]any{"text": p.Text})
			}
		case provider.PartReasoning:
			if p.Text != "" {
				// Bedrock requires signature for reasoning replay.
				// Skip reasoning blocks without signature to avoid validation errors.
				var sig string
				if p.ProviderOptions != nil {
					sig, _ = p.ProviderOptions["signature"].(string)
				}
				if sig == "" {
					continue
				}
				reasoningText := map[string]any{
					"text":      p.Text,
					"signature": sig,
				}
				blocks = append(blocks, map[string]any{"reasoningContent": map[string]any{
					"reasoningText": reasoningText,
				}})
			} else if p.ProviderOptions != nil {
				// Redacted reasoning (no text, just encrypted data).
				if data, ok := p.ProviderOptions["redactedData"].(string); ok && data != "" {
					blocks = append(blocks, map[string]any{"reasoningContent": map[string]any{
						"redactedReasoning": map[string]any{"data": data},
					}})
				}
			}
		case provider.PartImage:
			mediaType, data, ok := httpc.ParseDataURL(p.URL)
			format := "png"
			if p.MediaType != "" {
				if idx := strings.Index(p.MediaType, "/"); idx >= 0 {
					format = p.MediaType[idx+1:]
				}
			} else if ok {
				if idx := strings.Index(mediaType, "/"); idx >= 0 {
					format = mediaType[idx+1:]
				}
			}

			if !ok {
				data = p.URL
			}

			block := map[string]any{
				"image": map[string]any{
					"format": format,
					"source": map[string]any{"bytes": data},
				},
			}
			blocks = append(blocks, block)
		case provider.PartToolCall:
			var input any
			if len(p.ToolInput) > 0 {
				_ = json.Unmarshal(p.ToolInput, &input) // nil on error → fallback below
			}
			if input == nil {
				input = map[string]any{}
			}
			blocks = append(blocks, map[string]any{
				"toolUse": map[string]any{
					"toolUseId": p.ToolCallID,
					"name":      p.ToolName,
					"input":     input,
				},
			})
		case provider.PartToolResult:
			resultContent := []map[string]any{{"text": p.ToolOutput}}
			blocks = append(blocks, map[string]any{
				"toolResult": map[string]any{
					"toolUseId": p.ToolCallID,
					"content":   resultContent,
				},
			})
		case provider.PartFile:
			format := bedrockDocumentFormat(p.MediaType)
			name := sanitizeDocumentName(p.Filename)
			// Bedrock requires unique document names across the conversation.
			// Append "-N" suffix for duplicates.
			docNames[name]++
			if docNames[name] > 1 {
				name = fmt.Sprintf("%s-%d", name, docNames[name])
			}
			// p.URL may be a data URL ("data:mime;base64,...") or raw base64.
			// Bedrock Converse API expects raw base64 in source.bytes.
			_, data, _ := httpc.ParseDataURL(p.URL)
			if data == "" {
				data = p.URL
			}
			blocks = append(blocks, map[string]any{
				"document": map[string]any{
					"format": format,
					"name":   name,
					"source": map[string]any{"bytes": data},
				},
			})
		}

		// Cache point support: if part has CacheControl set, append a cachePoint block.
		if p.CacheControl != "" {
			blocks = append(blocks, map[string]any{
				"cachePoint": map[string]any{"type": "default"},
			})
		}
	}

	return blocks
}

// parseConverseResponse parses a non-streaming Converse API response.
func parseConverseResponse(body []byte) (*provider.GenerateResult, error) {
	var resp struct {
		Output struct {
			Message struct {
				Role    string            `json:"role"`
				Content []json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"output"`
		StopReason                    string          `json:"stopReason"`
		AdditionalModelResponseFields json.RawMessage `json:"additionalModelResponseFields,omitempty"`
		Trace                         json.RawMessage `json:"trace,omitempty"`
		Usage                         struct {
			InputTokens           int `json:"inputTokens"`
			OutputTokens          int `json:"outputTokens"`
			TotalTokens           int `json:"totalTokens"`
			CacheReadInputTokens  int `json:"cacheReadInputTokens"`
			CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing bedrock response: %w", err)
	}

	netInput := resp.Usage.InputTokens - resp.Usage.CacheReadInputTokens
	if netInput < 0 {
		netInput = 0
	}
	result := &provider.GenerateResult{
		Usage: provider.Usage{
			InputTokens:      netInput,
			OutputTokens:     resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
			CacheReadTokens:  resp.Usage.CacheReadInputTokens,
			CacheWriteTokens: resp.Usage.CacheWriteInputTokens,
		},
		FinishReason: mapStopReason(resp.StopReason),
	}

	// Parse content blocks.
	var reasoningParts []map[string]any
	var reasoningTextBuf strings.Builder
	for _, raw := range resp.Output.Message.Content {
		var block map[string]any
		if err := json.Unmarshal(raw, &block); err != nil {
			continue
		}
		if text, ok := block["text"].(string); ok {
			result.Text += text
		}
		if tu, ok := block["toolUse"].(map[string]any); ok {
			tc := provider.ToolCall{
				ID:   strVal(tu, "toolUseId"),
				Name: strVal(tu, "name"),
			}
			if input, exists := tu["input"]; exists {
				// json.Marshal cannot fail on values from json.Unmarshal.
				tc.Input, _ = json.Marshal(input)
			}
			result.ToolCalls = append(result.ToolCalls, tc)
		}
		if rc, ok := block["reasoningContent"].(map[string]any); ok {
			// Signed reasoning text.
			if rt, ok := rc["reasoningText"].(map[string]any); ok {
				part := map[string]any{"type": "reasoning"}
				if text, ok := rt["text"].(string); ok {
					part["text"] = text
					reasoningTextBuf.WriteString(text)
				}
				if sig, ok := rt["signature"].(string); ok {
					part["signature"] = sig
				}
				reasoningParts = append(reasoningParts, part)
			}
			// Redacted reasoning.
			if rd, ok := rc["redactedReasoning"].(map[string]any); ok {
				part := map[string]any{"type": "redacted_reasoning"}
				if data, ok := rd["data"].(string); ok {
					part["data"] = data
				}
				reasoningParts = append(reasoningParts, part)
			}
		}
	}

	// Surface concatenated reasoning text on the top-level result so
	// callers (goai TextResult.Reasoning) can render thinking in the
	// non-streaming path. Signatures + redacted blocks remain in
	// ProviderMetadata for replay.
	if reasoningTextBuf.Len() > 0 {
		result.Reasoning = reasoningTextBuf.String()
	}

	// Store reasoning metadata and additional response fields in ProviderMetadata.
	bedrockMeta := make(map[string]any)
	if len(reasoningParts) > 0 {
		bedrockMeta["reasoning"] = reasoningParts
	}
	if len(resp.AdditionalModelResponseFields) > 0 {
		var additional map[string]any
		if err := json.Unmarshal(resp.AdditionalModelResponseFields, &additional); err == nil {
			bedrockMeta["additionalModelResponseFields"] = additional
		}
	}
	if len(resp.Trace) > 0 {
		var trace any
		if err := json.Unmarshal(resp.Trace, &trace); err == nil {
			bedrockMeta["trace"] = trace
		}
	}
	if len(bedrockMeta) > 0 {
		result.ProviderMetadata = map[string]map[string]any{
			"bedrock": bedrockMeta,
		}
	}

	return result, nil
}

// parseEventStream parses a Bedrock Converse streaming response (EventStream binary protocol).
// When rfMode is true, a synthetic __json_response tool call is converted to a text chunk
// instead of being emitted as a tool call, mirroring the DoGenerate ResponseFormat behavior.
func parseEventStream(ctx context.Context, body io.ReadCloser, out chan<- provider.StreamChunk, responseMeta provider.ResponseMetadata, rfMode bool) {
	defer close(out)
	defer func() { _ = body.Close() }()

	// Watch for context cancellation and close the body to unblock decoder.Next().
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = body.Close()
		case <-done:
		}
	}()
	defer close(done)

	decoder := newEventStreamDecoder(body)

	// Track content blocks by index to know if a block is text or toolUse.
	// For tool blocks, accumulate input deltas so the final ChunkToolCall
	// includes the full tool input (matches openaicompat accumulation pattern).
	// isRFBlock marks the synthetic __json_response tool block used by rfMode.
	type blockInfo struct {
		isToolUse bool
		isRFBlock bool // true when rfMode and this is the synthetic RF tool
		toolUseID string
		toolName  string
		inputBuf  strings.Builder // accumulated tool input JSON fragments
	}
	blocks := make(map[int]*blockInfo)

	var finishSent bool
	var accUsage provider.Usage

	defer func() {
		// Fallback: if the stream ended without a metadata frame (e.g. premature
		// connection close), emit a ChunkFinish so consumers always see one.
		if !finishSent {
			provider.TrySend(ctx, out, provider.StreamChunk{
				Type:     provider.ChunkFinish,
				Usage:    accUsage,
				Response: responseMeta,
			})
		}
	}()

	for {
		frame, err := decoder.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: err}) {
				return
			}
			return
		}

		// Skip non-event frames (e.g., exceptions).
		if frame.MessageType == "exception" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:  provider.ChunkError,
				Error: fmt.Errorf("bedrock: %s: %s", frame.EventType, string(frame.Payload)),
			}) {
				return
			}
			return
		}
		if frame.MessageType != "event" {
			continue
		}

		var payload map[string]any
		if len(frame.Payload) > 0 {
			if err := json.Unmarshal(frame.Payload, &payload); err != nil {
				continue
			}
		}

		switch frame.EventType {
		case "contentBlockStart":
			idx := intVal(payload, "contentBlockIndex")
			if start, ok := payload["start"].(map[string]any); ok {
				if tu, ok := start["toolUse"].(map[string]any); ok {
					toolName := strVal(tu, "name")
					isRF := rfMode && toolName == responseFormatToolName
					bi := &blockInfo{
						isToolUse: true,
						isRFBlock: isRF,
						toolUseID: strVal(tu, "toolUseId"),
						toolName:  toolName,
					}
					blocks[idx] = bi
					// In rfMode the RF tool block is hidden from callers; only
					// emit ChunkToolCallStreamStart for real tool calls.
					if !isRF {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type:       provider.ChunkToolCallStreamStart,
							ToolCallID: bi.toolUseID,
							ToolName:   bi.toolName,
						}) {
							return
						}
					}
				} else {
					blocks[idx] = &blockInfo{isToolUse: false}
				}
			}

		case "contentBlockDelta":
			idx := intVal(payload, "contentBlockIndex")
			if delta, ok := payload["delta"].(map[string]any); ok {
				if text, ok := delta["text"].(string); ok {
					if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: text}) {
						return
					}
				}
				if tu, ok := delta["toolUse"].(map[string]any); ok {
					bi := blocks[idx]
					inputFrag := strVal(tu, "input")
					if bi != nil {
						bi.inputBuf.WriteString(inputFrag)
					}
					// RF blocks are converted to text at contentBlockStop; suppress
					// intermediate tool deltas so callers never see the synthetic tool.
					if bi == nil || !bi.isRFBlock {
						chunk := provider.StreamChunk{
							Type:      provider.ChunkToolCallDelta,
							ToolInput: inputFrag,
						}
						if bi != nil {
							chunk.ToolCallID = bi.toolUseID
							chunk.ToolName = bi.toolName
						}
						if !provider.TrySend(ctx, out, chunk) {
							return
						}
					}
				}
				if rc, ok := delta["reasoningContent"].(map[string]any); ok {
					// Streaming format: reasoningContent.text (string) for text,
					// reasoningContent.signature (string) for signature.
					// Non-streaming format: reasoningContent.reasoningText.{text, signature}.
					if text, ok := rc["text"].(string); ok {
						if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: text}) {
							return
						}
					}
					if sig, ok := rc["signature"].(string); ok && sig != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type:     provider.ChunkReasoning,
							Metadata: map[string]any{"signature": sig},
						}) {
							return
						}
					}
					// Non-streaming format fallback: reasoningText wrapper.
					if rt, ok := rc["reasoningText"].(map[string]any); ok {
						chunk := provider.StreamChunk{Type: provider.ChunkReasoning, Text: strVal(rt, "text")}
						if sig := strVal(rt, "signature"); sig != "" {
							chunk.Metadata = map[string]any{"signature": sig}
						}
						if !provider.TrySend(ctx, out, chunk) {
							return
						}
					}
					// Redacted reasoning.
					if rd, ok := rc["redactedReasoning"].(map[string]any); ok {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type:     provider.ChunkReasoning,
							Metadata: map[string]any{"redactedData": strVal(rd, "data")},
						}) {
							return
						}
					}
				}
			}

		case "contentBlockStop":
			idx := intVal(payload, "contentBlockIndex")
			if bi, ok := blocks[idx]; ok && bi.isToolUse {
				if bi.isRFBlock {
					// Convert the RF tool input to a text chunk.
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type: provider.ChunkText,
						Text: bi.inputBuf.String(),
					}) {
						return
					}
				} else {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCall,
						ToolCallID: bi.toolUseID,
						ToolName:   bi.toolName,
						ToolInput:  bi.inputBuf.String(),
					}) {
						return
					}
				}
			}

		case "messageStop":
			stopReason := strVal(payload, "stopReason")
			fr := mapStopReason(stopReason)
			// In rfMode the RF tool_use finish is semantically a stop.
			if rfMode && fr == provider.FinishToolCalls {
				fr = provider.FinishStop
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: fr,
			}) {
				return
			}

		case "metadata":
			var usage provider.Usage
			if u, ok := payload["usage"].(map[string]any); ok {
				usage.InputTokens = intVal(u, "inputTokens")
				usage.OutputTokens = intVal(u, "outputTokens")
				usage.TotalTokens = intVal(u, "totalTokens")
				usage.CacheReadTokens = intVal(u, "cacheReadInputTokens")
				usage.CacheWriteTokens = intVal(u, "cacheWriteInputTokens")
				// InputTokens from Bedrock includes cache-read tokens; subtract to
				// report only net (uncached) input tokens, matching non-streaming behavior.
				usage.InputTokens -= usage.CacheReadTokens
				if usage.InputTokens < 0 {
					usage.InputTokens = 0
				}
			}
			accUsage = usage
			chunk := provider.StreamChunk{
				Type:     provider.ChunkFinish,
				Usage:    usage,
				Response: responseMeta,
			}
			// Include cache write tokens in metadata for consumers that need it.
			if usage.CacheWriteTokens > 0 {
				chunk.Metadata = map[string]any{
					"cacheWriteInputTokens": usage.CacheWriteTokens,
				}
			}
			finishSent = true
			if !provider.TrySend(ctx, out, chunk) {
				return
			}
		}
	}
}

// mapStopReason maps Bedrock stop reasons to provider.FinishReason.
// Matches Vercel AI SDK's BEDROCK_STOP_REASONS mapping.
func mapStopReason(reason string) provider.FinishReason {
	switch reason {
	case "", "end_turn", "stop", "stop_sequence":
		return provider.FinishStop
	case "tool_use", "tool-calls":
		return provider.FinishToolCalls
	case "max_tokens", "length":
		return provider.FinishLength
	case "content_filtered", "content-filter", "guardrail_intervened":
		return provider.FinishContentFilter
	default:
		return provider.FinishOther
	}
}

func strVal(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func intVal(m map[string]any, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}

// bedrockDocumentFormat maps a MIME type to the Bedrock document format string.
// Matches Vercel AI SDK's getBedrockDocumentFormat().
func bedrockDocumentFormat(mimeType string) string {
	switch mimeType {
	case "application/pdf":
		return "pdf"
	case "text/csv":
		return "csv"
	case "application/msword":
		return "doc"
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return "docx"
	case "application/vnd.ms-excel":
		return "xls"
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return "xlsx"
	case "text/html":
		return "html"
	case "text/plain":
		return "txt"
	case "text/markdown":
		return "md"
	default:
		return "txt"
	}
}

// ensureToolResultPairing moved to provider.NormalizeToolMessages (provider/normalize.go).
