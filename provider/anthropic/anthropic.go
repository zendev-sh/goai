// Package anthropic provides an Anthropic language model implementation for GoAI.
//
// It uses the Anthropic Messages API with native SSE streaming.
//
// Usage:
//
//	model := anthropic.Chat("claude-sonnet-4-20250514", anthropic.WithAPIKey("sk-ant-..."))
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package anthropic

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

const (
	defaultBaseURL    = "https://api.anthropic.com"
	apiVersion        = "2023-06-01"
	betaFeatures      = "claude-code-20250219,interleaved-thinking-2025-05-14"
	defaultMaxTokens  = 16384
)

// Option configures the Anthropic provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.tokenSource = provider.StaticToken(key)
	}
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) {
		o.tokenSource = ts
	}
}

// WithBaseURL overrides the default Anthropic API base URL.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.baseURL = url
	}
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(h map[string]string) Option {
	return func(o *options) {
		o.headers = h
	}
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

// Chat creates an Anthropic language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	if o.tokenSource == nil {
		if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("ANTHROPIC_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return &chatModel{
		id:   modelID,
		opts: o,
	}
}

type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

// supportsThinking returns true for Anthropic models that support extended thinking.
func supportsThinking(modelID string) bool {
	return strings.Contains(modelID, "claude-3-7-sonnet") ||
		strings.Contains(modelID, "claude-sonnet-4") ||
		strings.Contains(modelID, "claude-opus-4")
}

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: true,
		Reasoning:   supportsThinking(m.id),
		ToolCall:    true,
		Attachment:  true,
		InputModalities: provider.ModalitySet{
			Text:  true,
			Image: true,
			PDF:   true,
		},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	useOutputFormat := m.useNativeOutputFormat(params)
	rfMode := params.ResponseFormat != nil && !useOutputFormat
	if rfMode {
		params = injectResponseFormatTool(params)
	} else if useOutputFormat {
		params = injectNativeOutputFormat(params)
	}
	body := m.buildRequest(params, false)
	toolBetas := collectToolBetas(params.Tools)

	resp, err := m.doHTTP(ctx, body, toolBetas...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	result, err := parseResponse(respBody)
	if err != nil {
		return nil, err
	}

	if rfMode {
		extractResponseFormatResult(result)
	}
	return result, nil
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	useOutputFormat := m.useNativeOutputFormat(params)
	rfMode := params.ResponseFormat != nil && !useOutputFormat
	if rfMode {
		params = injectResponseFormatTool(params)
	} else if useOutputFormat {
		params = injectNativeOutputFormat(params)
	}
	body := m.buildRequest(params, true)
	toolBetas := collectToolBetas(params.Tools)

	resp, err := m.doHTTP(ctx, body, toolBetas...)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				_ = resp.Body.Close()
			case <-done:
			}
		}()
		parseSSE(ctx, resp.Body, out, rfMode)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

// --- Request building ---

func (m *chatModel) buildRequest(params provider.GenerateParams, streaming bool) map[string]any {
	body := map[string]any{
		"model":      m.id,
		"stream":     streaming,
		"max_tokens": m.maxTokens(params),
	}

	// System prompt as content array.
	// Cache control is conditional -- only applied when consumer enables PromptCaching,
	// matching Vercel AI SDK's CacheControlValidator pattern.
	if params.System != "" {
		systemPart := map[string]any{"type": "text", "text": params.System}
		if params.PromptCaching {
			systemPart["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		body["system"] = []map[string]any{systemPart}
	}

	// Messages.
	body["messages"] = convertMessages(params.Messages)

	// Tools.
	if len(params.Tools) > 0 {
		tools := make([]map[string]any, len(params.Tools))
		for i, t := range params.Tools {
			tools[i] = convertToolToAPI(t)
		}
		body["tools"] = tools
	}

	// Tool choice.
	if params.ToolChoice != "" {
		switch params.ToolChoice {
		case "auto":
			body["tool_choice"] = map[string]any{"type": "auto"}
		case "none":
			// Anthropic doesn't have a "none" tool_choice; omit tools instead.
			delete(body, "tools")
		case "required":
			body["tool_choice"] = map[string]any{"type": "any"}
		default:
			body["tool_choice"] = map[string]any{"type": "tool", "name": params.ToolChoice}
		}
	}

	// Temperature.
	if params.Temperature != nil {
		body["temperature"] = *params.Temperature
	}

	// TopP.
	if params.TopP != nil {
		body["top_p"] = *params.TopP
	}

	// TopK.
	if params.TopK != nil {
		body["top_k"] = *params.TopK
	}

	// Stop sequences.
	if len(params.StopSequences) > 0 {
		body["stop_sequences"] = params.StopSequences
	}

	// Thinking / extended thinking.
	// Read from ProviderOptions["thinking"] -- matches Vercel AI SDK convention.
	// Accepts: {type: "enabled", budgetTokens: N} or {type: "adaptive"} or {type: "disabled"}.
	if thinking, ok := params.ProviderOptions["thinking"]; ok {
		if tm, ok := thinking.(map[string]any); ok {
			thinkingReq := map[string]any{}
			if t, ok := tm["type"]; ok {
				thinkingReq["type"] = t
			}
			if budget, ok := tm["budgetTokens"]; ok {
				thinkingReq["budget_tokens"] = budget
			}
			if len(thinkingReq) > 0 {
				body["thinking"] = thinkingReq
			}
		}
	}

	// disableParallelToolUse -- add disable_parallel_tool_use to tool_choice.
	// Matches Vercel AI SDK: when set, adds the field to any tool_choice variant.
	if disable, ok := params.ProviderOptions["disableParallelToolUse"]; ok {
		if b, ok := disable.(bool); ok && b {
			if tc, ok := body["tool_choice"].(map[string]any); ok {
				tc["disable_parallel_tool_use"] = true
			} else if len(params.Tools) > 0 && params.ToolChoice == "" {
				// Default tool_choice is auto when tools are present.
				body["tool_choice"] = map[string]any{
					"type":                       "auto",
					"disable_parallel_tool_use": true,
				}
			}
		}
	}

	// effort -- output quality level (low/medium/high/max).
	// Vercel wraps this as output_config.effort.
	if effort, ok := params.ProviderOptions["effort"]; ok {
		if e, ok := effort.(string); ok && e != "" {
			body["output_config"] = map[string]any{"effort": e}
		}
	}

	// speed -- fast/standard inference speed.
	if speed, ok := params.ProviderOptions["speed"]; ok {
		if s, ok := speed.(string); ok && s != "" {
			body["speed"] = s
		}
	}

	// container -- code execution container specification.
	if container, ok := params.ProviderOptions["container"]; ok {
		if cm, ok := container.(map[string]any); ok {
			apiContainer := map[string]any{}
			if id, ok := cm["id"]; ok {
				apiContainer["id"] = id
			}
			if skills, ok := cm["skills"]; ok {
				if skillList, ok := skills.([]any); ok {
					apiSkills := make([]map[string]any, 0, len(skillList))
					for _, s := range skillList {
						if sm, ok := s.(map[string]any); ok {
							skill := map[string]any{}
							if t, ok := sm["type"]; ok {
								skill["type"] = t
							}
							if sid, ok := sm["skillId"]; ok {
								skill["skill_id"] = sid
							}
							if v, ok := sm["version"]; ok {
								skill["version"] = v
							}
							apiSkills = append(apiSkills, skill)
						}
					}
					apiContainer["skills"] = apiSkills
				}
			}
			body["container"] = apiContainer
		}
	}

	// contextManagement -- automatic context window management.
	if cm, ok := params.ProviderOptions["contextManagement"]; ok {
		if cmm, ok := cm.(map[string]any); ok {
			if edits, ok := cmm["edits"]; ok {
				if editList, ok := edits.([]any); ok {
					apiEdits := make([]map[string]any, 0, len(editList))
					for _, e := range editList {
						if em, ok := e.(map[string]any); ok {
							apiEdit := map[string]any{}
							if t, ok := em["type"]; ok {
								apiEdit["type"] = t
							}
							// Pass through all sub-fields with snake_case conversion.
							for k, v := range em {
								switch k {
								case "type":
									// already handled
								case "trigger", "keep":
									apiEdit[k] = v
								case "clearAtLeast":
									apiEdit["clear_at_least"] = v
								case "clearToolInputs":
									apiEdit["clear_tool_inputs"] = v
								case "excludeTools":
									apiEdit["exclude_tools"] = v
								case "pauseAfterCompaction":
									apiEdit["pause_after_compaction"] = v
								case "instructions":
									apiEdit["instructions"] = v
								default:
									apiEdit[k] = v
								}
							}
							apiEdits = append(apiEdits, apiEdit)
						}
					}
					body["context_management"] = map[string]any{"edits": apiEdits}
				}
			}
		}
	}

	// structuredOutputMode -- native output_format for newer Claude models.
	// When "outputFormat" or "auto" (with supported model), use output_format instead of tool trick.
	// This is checked here for the passthrough case; the main ResponseFormat logic
	// is handled in injectResponseFormatTool / the caller.

	// Provider options passthrough -- allows callers to inject arbitrary
	// request body fields. Skip keys handled above.
	handledKeys := map[string]bool{
		"thinking": true, "_headers": true,
		"disableParallelToolUse": true, "effort": true, "speed": true,
		"container": true, "contextManagement": true,
	}
	protectedKeys := map[string]bool{
		"model": true, "stream": true, "messages": true,
		"max_tokens": true, "system": true, "temperature": true,
		"top_p": true, "top_k": true, "stop_sequences": true,
		"tools": true, "tool_choice": true,
		// SDK-internal keys that are never sent on the wire.
		"structuredOutputMode": true, "sendReasoning": true,
		"cacheControl": true,
	}
	for k, v := range params.ProviderOptions {
		if handledKeys[k] || protectedKeys[k] {
			continue
		}
		body[k] = v
	}

	// Per-request headers (extracted in doHTTP before marshaling).
	if len(params.Headers) > 0 {
		body["_headers"] = params.Headers
	}

	return body
}

func (m *chatModel) maxTokens(params provider.GenerateParams) int {
	if params.MaxOutputTokens > 0 {
		return params.MaxOutputTokens
	}
	return defaultMaxTokens
}

// --- Message conversion ---

func convertMessages(msgs []provider.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))
	for _, msg := range msgs {
		if msg.Role == provider.RoleSystem {
			continue // system handled separately
		}

		role := string(msg.Role)
		if msg.Role == provider.RoleTool {
			role = "user"
		}
		m := map[string]any{"role": role}
		content := make([]map[string]any, 0, len(msg.Content))

		// Check message-level cache control from ProviderOptions.
		var msgCacheControl map[string]any
		if anthropicOpts, ok := msg.ProviderOptions["anthropic"].(map[string]any); ok {
			if cc, ok := anthropicOpts["cacheControl"].(map[string]any); ok {
				msgCacheControl = cc
			}
		}

		for i, part := range msg.Content {
			isLast := i == len(msg.Content)-1

			switch part.Type {
			case provider.PartText:
				if part.Text == "" {
					continue
				}
				p := map[string]any{"type": "text", "text": part.Text}
				applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartReasoning:
				if part.Text != "" {
					// Signature is required for replaying thinking blocks.
					// Skip reasoning from other providers (e.g. Gemini) that lack signatures.
					var sig string
					if part.ProviderOptions != nil {
						sig, _ = part.ProviderOptions["signature"].(string)
					}
					if sig == "" {
						continue
					}
					p := map[string]any{"type": "thinking", "thinking": part.Text, "signature": sig}
					applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
					content = append(content, p)
				} else if part.ProviderOptions != nil {
					// Redacted thinking (no text, just encrypted data).
					if data, ok := part.ProviderOptions["redactedData"].(string); ok && data != "" {
						p := map[string]any{"type": "redacted_thinking", "data": data}
						content = append(content, p)
					}
				}

			case provider.PartImage:
				if part.URL == "" {
					continue
				}
				mediaType, data, ok := httpc.ParseDataURL(part.URL)
				if !ok {
					continue
				}
				p := map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       "base64",
						"media_type": mediaType,
						"data":       data,
					},
				}
				applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartFile:
				if part.URL == "" {
					continue
				}
				mediaType, data, ok := httpc.ParseDataURL(part.URL)
				if !ok {
					continue
				}
				p := map[string]any{
					"type": "document",
					"source": map[string]any{
						"type":       "base64",
						"media_type": mediaType,
						"data":       data,
					},
				}
				applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartToolCall:
				var input any
				if len(part.ToolInput) > 0 {
					if err := json.Unmarshal(part.ToolInput, &input); err != nil {
						input = map[string]any{}
					}
				}
				if input == nil {
					input = map[string]any{}
				}
				p := map[string]any{
					"type":  "tool_use",
					"id":    part.ToolCallID,
					"name":  part.ToolName,
					"input": input,
				}
				applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartToolResult:
				p := map[string]any{
					"type":        "tool_result",
					"tool_use_id": part.ToolCallID,
					"content":     part.ToolOutput,
				}
				applyCacheControl(p, part.CacheControl, msgCacheControl, isLast)
				content = append(content, p)
			}
		}

		// Anthropic rejects messages with empty content arrays.
		if len(content) == 0 {
			continue
		}

		m["content"] = content
		result = append(result, m)
	}
	return result
}

// applyCacheControl adds cache_control to a content part.
// Part-level CacheControl takes precedence; message-level only applies to the last part.
func applyCacheControl(p map[string]any, partCC string, msgCC map[string]any, isLast bool) {
	if partCC != "" {
		p["cache_control"] = map[string]any{"type": partCC}
	} else if msgCC != nil && isLast {
		p["cache_control"] = msgCC
	}
}

// --- Response format (structured output via tool trick or native output_format) ---

// useNativeOutputFormat checks if the caller requested native output_format mode
// via ProviderOptions["structuredOutputMode"] = "outputFormat" (or "auto" with a supported model).
func (m *chatModel) useNativeOutputFormat(params provider.GenerateParams) bool {
	mode, _ := params.ProviderOptions["structuredOutputMode"].(string)
	switch mode {
	case "outputFormat":
		return true
	case "auto":
		return m.supportsNativeOutputFormat()
	default:
		return false
	}
}

// supportsNativeOutputFormat returns true for models that support native output_format.
// Matches Vercel: claude-sonnet-4-6, claude-opus-4-6, claude-sonnet-4-5, claude-opus-4-5, claude-opus-4-1.
func (m *chatModel) supportsNativeOutputFormat() bool {
	id := m.id
	return strings.Contains(id, "claude-sonnet-4-6") ||
		strings.Contains(id, "claude-opus-4-6") ||
		strings.Contains(id, "claude-sonnet-4-5") ||
		strings.Contains(id, "claude-opus-4-5") ||
		strings.Contains(id, "claude-opus-4-1")
}

// injectNativeOutputFormat adds output_format to ProviderOptions for passthrough to the API body.
func injectNativeOutputFormat(params provider.GenerateParams) provider.GenerateParams {
	p := params
	// Copy the map to avoid mutating the caller's ProviderOptions.
	newOpts := maps.Clone(p.ProviderOptions)
	if newOpts == nil {
		newOpts = make(map[string]any, 1)
	}
	p.ProviderOptions = newOpts
	if p.ResponseFormat == nil {
		return params
	}
	var schema any
	if len(p.ResponseFormat.Schema) > 0 {
		if err := json.Unmarshal(p.ResponseFormat.Schema, &schema); err != nil {
			return params // schema invalid, fall back to tool trick
		}
	}
	p.ProviderOptions["output_format"] = map[string]any{
		"type":   "json_schema",
		"schema": schema,
	}
	// Clear ResponseFormat so the tool trick is not also applied.
	p.ResponseFormat = nil
	return p
}

const responseFormatToolName = "json_response"

// injectResponseFormatTool adds a synthetic tool to force JSON output via tool_use.
func injectResponseFormatTool(params provider.GenerateParams) provider.GenerateParams {
	p := params
	p.Tools = append([]provider.ToolDefinition{{
		Name:        responseFormatToolName,
		Description: "Return structured JSON response",
		InputSchema: params.ResponseFormat.Schema,
	}}, p.Tools...)
	p.ToolChoice = responseFormatToolName
	return p
}

// extractResponseFormatResult converts the synthetic tool call result to text.
func extractResponseFormatResult(result *provider.GenerateResult) {
	for i, tc := range result.ToolCalls {
		if tc.Name == responseFormatToolName {
			result.Text = string(tc.Input)
			// Remove the synthetic tool call from the list.
			result.ToolCalls = append(result.ToolCalls[:i], result.ToolCalls[i+1:]...)
			if len(result.ToolCalls) == 0 {
				result.FinishReason = provider.FinishStop
			}
			return
		}
	}
}

// --- SSE parsing ---

func parseSSE(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk, isRFMode bool) {
	defer close(out)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var currentToolCallID string
	var currentToolName string
	var currentToolArgs strings.Builder // accumulate partial JSON fragments
	var isRFBlock bool                  // true when current tool_use block is the synthetic response format tool
	var isFirstDelta bool              // true for first input_json_delta of a tool_use block
	var isServerTool bool              // true when current block is server_tool_use
	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	var finishMeta map[string]any      // metadata accumulated for ChunkFinish

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		eventType, _ := event["type"].(string)

		switch eventType {
		case "message_start":
			if msg, ok := event["message"].(map[string]any); ok {
				if id, ok := msg["id"].(string); ok {
					responseMeta.ID = id
				}
				if model, ok := msg["model"].(string); ok {
					responseMeta.Model = model
				}
				if u, ok := msg["usage"].(map[string]any); ok {
					if v, ok := u["input_tokens"].(float64); ok {
						usage.InputTokens = int(v)
					}
					if v, ok := u["cache_read_input_tokens"].(float64); ok {
						usage.CacheReadTokens = int(v)
					}
					if v, ok := u["cache_creation_input_tokens"].(float64); ok {
						usage.CacheWriteTokens = int(v)
					}
				}
			}

		case "content_block_start":
			if cb, ok := event["content_block"].(map[string]any); ok {
				cbType, _ := cb["type"].(string)
				switch cbType {
				case "tool_use":
					currentToolCallID, _ = cb["id"].(string)
					currentToolName, _ = cb["name"].(string)
					isRFBlock = isRFMode && currentToolName == responseFormatToolName
					isFirstDelta = true
					isServerTool = false
					if !isRFBlock {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type:       provider.ChunkToolCallStreamStart,
							ToolCallID: currentToolCallID,
							ToolName:   currentToolName,
						}) {
							return
						}
					}
				case "server_tool_use":
					currentToolCallID, _ = cb["id"].(string)
					currentToolName, _ = cb["name"].(string)
					isRFBlock = false
					isFirstDelta = true
					isServerTool = true
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCallStreamStart,
						ToolCallID: currentToolCallID,
						ToolName:   currentToolName,
					}) {
						return
					}
				}
			}

		case "content_block_delta":
			if delta, ok := event["delta"].(map[string]any); ok {
				deltaType, _ := delta["type"].(string)
				switch deltaType {
				case "text_delta":
					text, _ := delta["text"].(string)
					if text != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: text}) {
							return
						}
					}
				case "thinking_delta":
					text, _ := delta["thinking"].(string)
					if text != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: text}) {
							return
						}
					}
				case "signature_delta":
					sig, _ := delta["signature"].(string)
					if sig != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type: provider.ChunkReasoning,
							Text: "",
							Metadata: map[string]any{
								"signature": sig,
							},
						}) {
							return
						}
					}
				case "citations_delta":
					if citation, ok := delta["citation"].(map[string]any); ok {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type: provider.ChunkText,
							Text: "",
							Metadata: map[string]any{
								"citation": citation,
							},
						}) {
							return
						}
					}
				case "input_json_delta":
					text, _ := delta["partial_json"].(string)
					if text != "" {
						if isRFBlock {
							// In response format mode: emit tool input as text chunks.
							if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: text}) {
								return
							}
						} else {
							// CRITICAL: firstDelta JSON wrapping for code execution tools.
							// When server_tool_use subtypes (bash_code_execution, text_editor_code_execution)
							// are split out, the API sends input without the "type" field.
							// We inject it back, matching Vercel's behavior.
							emitText := text
							if isFirstDelta && isServerTool {
								if currentToolName == "bash_code_execution" || currentToolName == "text_editor_code_execution" {
									emitText = `{"type": "` + currentToolName + `",` + text[1:]
								}
							}
							isFirstDelta = false
							// Accumulate partial JSON fragments; emit complete JSON on content_block_stop.
							currentToolArgs.WriteString(emitText)
							// Emit delta for UI streaming progress (matches Vercel's tool-input-delta).
							if !provider.TrySend(ctx, out, provider.StreamChunk{
								Type:       provider.ChunkToolCallDelta,
								ToolCallID: currentToolCallID,
								ToolName:   currentToolName,
								ToolInput:  emitText,
							}) {
								return
							}
						}
					}
				}
			}

		case "content_block_stop":
			if currentToolCallID != "" && !isRFBlock {
				// Emit accumulated tool call with complete JSON args.
				args := cmp.Or(currentToolArgs.String(), "{}")
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCall,
					ToolCallID: currentToolCallID,
					ToolName:   currentToolName,
					ToolInput:  args,
				}) {
					return
				}
				currentToolArgs.Reset()
			}
			if currentToolCallID != "" {
				currentToolCallID = ""
				currentToolName = ""
				isRFBlock = false
				isServerTool = false
			}

		case "message_delta":
			if delta, ok := event["delta"].(map[string]any); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					fr := mapFinishReason(sr)
					// In RF mode, tool_use finish → stop (we consumed the tool as text).
					if isRFMode && fr == provider.FinishToolCalls {
						fr = provider.FinishStop
					}
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:         provider.ChunkStepFinish,
						FinishReason: fr,
					}) {
						return
					}
				}
				// Container from message_delta.
				if container, ok := delta["container"].(map[string]any); ok {
					if finishMeta == nil {
						finishMeta = map[string]any{}
					}
					finishMeta["container"] = container
				}
			}
			if u, ok := event["usage"].(map[string]any); ok {
				// Handle iterations -- sum across iterations for total usage.
				if iters, ok := u["iterations"].([]any); ok && len(iters) > 0 {
					totalIn, totalOut := 0, 0
					iterMeta := make([]map[string]any, 0, len(iters))
					for _, iter := range iters {
						if im, ok := iter.(map[string]any); ok {
							inTok, _ := im["input_tokens"].(float64)
							outTok, _ := im["output_tokens"].(float64)
							totalIn += int(inTok)
							totalOut += int(outTok)
							iterType, _ := im["type"].(string)
							iterMeta = append(iterMeta, map[string]any{
								"type":         iterType,
								"inputTokens":  int(inTok),
								"outputTokens": int(outTok),
							})
						}
					}
					usage.InputTokens = totalIn
					usage.OutputTokens = totalOut
					if finishMeta == nil {
						finishMeta = map[string]any{}
					}
					finishMeta["iterations"] = iterMeta
				} else {
					if v, ok := u["output_tokens"].(float64); ok {
						usage.OutputTokens = int(v)
					}
				}
				// Reasoning tokens from output_tokens_details.
				if details, ok := u["output_tokens_details"].(map[string]any); ok {
					if v, ok := details["thinking_tokens"].(float64); ok {
						usage.ReasoningTokens = int(v)
					}
				}
			}
			// Context management from message_delta.
			if cm, ok := event["context_management"].(map[string]any); ok {
				if finishMeta == nil {
					finishMeta = map[string]any{}
				}
				finishMeta["contextManagement"] = cm
			}

		case "message_stop":
			usage.TotalTokens = usage.InputTokens + usage.OutputTokens
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:     provider.ChunkFinish,
				Usage:    usage,
				Response: responseMeta,
				Metadata: finishMeta,
			}) {
				return
			}
			return

		case "error":
			handleStreamError(ctx, data, event, out)
			return
		}
	}

	if err := scanner.Err(); err != nil {
		if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) {
			return
		}
	}
}

func handleStreamError(ctx context.Context, data string, event map[string]any, out chan<- provider.StreamChunk) {
	// Try ClassifyStreamError for structured error detection.
	if streamErr := goai.ClassifyStreamError([]byte(data)); streamErr != nil {
		provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: streamErr})
		return
	}

	errObj, _ := event["error"].(map[string]any)
	msg, _ := errObj["message"].(string)
	msg = cmp.Or(msg, "unknown stream error")

	var chunk provider.StreamChunk
	if goai.IsOverflow(msg) {
		chunk = provider.StreamChunk{Type: provider.ChunkError, Error: &goai.ContextOverflowError{Message: msg, ResponseBody: data}}
	} else {
		chunk = provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg}}
	}
	provider.TrySend(ctx, out, chunk) // terminal send: function exits immediately
}

// mapFinishReason converts Anthropic stop reasons to GoAI FinishReason.
func mapFinishReason(reason string) provider.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return provider.FinishStop
	case "tool_use":
		return provider.FinishToolCalls
	case "max_tokens", "model_context_window_exceeded":
		return provider.FinishLength
	case "refusal":
		return provider.FinishContentFilter
	default:
		return provider.FinishOther
	}
}

// --- Non-streaming response parsing ---

func parseResponse(body []byte) (*provider.GenerateResult, error) {
	var resp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Type    string `json:"type"`
		Content []struct {
			Type      string          `json:"type"`
			Text      string          `json:"text,omitempty"`
			ID        string          `json:"id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Input     json.RawMessage `json:"input,omitempty"`
			Thinking  string          `json:"thinking,omitempty"`
			Signature string          `json:"signature,omitempty"`
			Data      string          `json:"data,omitempty"` // redacted_thinking
			Citations []struct {
				Type           string `json:"type"`
				CitedText      string `json:"cited_text"`
				URL            string `json:"url,omitempty"`
				Title          string `json:"title,omitempty"`
				EncryptedIndex string `json:"encrypted_index,omitempty"`
				DocumentIndex  int    `json:"document_index,omitempty"`
				DocumentTitle  *string `json:"document_title,omitempty"`
				StartPageNumber int   `json:"start_page_number,omitempty"`
				EndPageNumber   int   `json:"end_page_number,omitempty"`
				StartCharIndex  int   `json:"start_char_index,omitempty"`
				EndCharIndex    int   `json:"end_char_index,omitempty"`
			} `json:"citations,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      *struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			OutputTokensDetails      *struct {
				ThinkingTokens int `json:"thinking_tokens"`
			} `json:"output_tokens_details,omitempty"`
			Iterations []struct {
				Type         string `json:"type"`
				InputTokens  int    `json:"input_tokens"`
				OutputTokens int    `json:"output_tokens"`
			} `json:"iterations,omitempty"`
		} `json:"usage"`
		ContextManagement *struct {
			AppliedEdits []map[string]any `json:"applied_edits"`
		} `json:"context_management,omitempty"`
		Container *struct {
			ExpiresAt string `json:"expires_at"`
			ID        string `json:"id"`
			Skills    []struct {
				Type    string `json:"type"`
				SkillID string `json:"skill_id"`
				Version string `json:"version"`
			} `json:"skills,omitempty"`
		} `json:"container,omitempty"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing anthropic response: %w", err)
	}

	// Handle error response.
	if resp.Error != nil {
		if goai.IsOverflow(resp.Error.Message) {
			return nil, &goai.ContextOverflowError{Message: resp.Error.Message, ResponseBody: string(body)}
		}
		return nil, &goai.APIError{Message: resp.Error.Message}
	}
	if resp.Type == "error" {
		return nil, &goai.APIError{Message: "unknown error"}
	}

	result := &provider.GenerateResult{
		Response: provider.ResponseMetadata{
			ID:    resp.ID,
			Model: resp.Model,
		},
		FinishReason: mapFinishReason(resp.StopReason),
	}

	// Extract text, tool calls, reasoning, and citations.
	var textParts []string
	var providerMeta map[string]any
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
			// Extract citations from text blocks.
			if len(block.Citations) > 0 {
				if providerMeta == nil {
					providerMeta = map[string]any{}
				}
				citations := make([]map[string]any, len(block.Citations))
				for i, c := range block.Citations {
					cit := map[string]any{
						"type":      c.Type,
						"citedText": c.CitedText,
					}
					switch c.Type {
					case "web_search_result_location":
						cit["url"] = c.URL
						cit["title"] = c.Title
						cit["encryptedIndex"] = c.EncryptedIndex
					case "page_location":
						cit["documentIndex"] = c.DocumentIndex
						if c.DocumentTitle != nil {
							cit["documentTitle"] = *c.DocumentTitle
						}
						cit["startPageNumber"] = c.StartPageNumber
						cit["endPageNumber"] = c.EndPageNumber
					case "char_location":
						cit["documentIndex"] = c.DocumentIndex
						if c.DocumentTitle != nil {
							cit["documentTitle"] = *c.DocumentTitle
						}
						cit["startCharIndex"] = c.StartCharIndex
						cit["endCharIndex"] = c.EndCharIndex
					}
					citations[i] = cit
				}
				existingCitations, _ := providerMeta["citations"].([]map[string]any)
				providerMeta["citations"] = append(existingCitations, citations...)
			}
		case "thinking":
			if block.Thinking != "" {
				// Reasoning text is not appended to result.Text -- it's metadata.
				if providerMeta == nil {
					providerMeta = map[string]any{}
				}
				reasoning, _ := providerMeta["reasoning"].([]map[string]any)
				entry := map[string]any{"type": "thinking", "text": block.Thinking}
				if block.Signature != "" {
					entry["signature"] = block.Signature
				}
				providerMeta["reasoning"] = append(reasoning, entry)
			}
		case "redacted_thinking":
			if providerMeta == nil {
				providerMeta = map[string]any{}
			}
			reasoning, _ := providerMeta["reasoning"].([]map[string]any)
			providerMeta["reasoning"] = append(reasoning, map[string]any{
				"type": "redacted_thinking", "data": block.Data,
			})
		case "tool_use", "server_tool_use":
			result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
				ID:    block.ID,
				Name:  block.Name,
				Input: block.Input,
			})
		}
	}
	result.Text = strings.Join(textParts, "")

	// Usage.
	if resp.Usage != nil {
		// When iterations are present, sum across iterations for total usage.
		inputTokens := resp.Usage.InputTokens
		outputTokens := resp.Usage.OutputTokens
		if len(resp.Usage.Iterations) > 0 {
			totalIn, totalOut := 0, 0
			for _, iter := range resp.Usage.Iterations {
				totalIn += iter.InputTokens
				totalOut += iter.OutputTokens
			}
			inputTokens = totalIn
			outputTokens = totalOut
		}
		result.Usage = provider.Usage{
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			TotalTokens:      inputTokens + outputTokens,
			CacheReadTokens:  resp.Usage.CacheReadInputTokens,
			CacheWriteTokens: resp.Usage.CacheCreationInputTokens,
		}
		// Reasoning tokens.
		if resp.Usage.OutputTokensDetails != nil {
			result.Usage.ReasoningTokens = resp.Usage.OutputTokensDetails.ThinkingTokens
		}
		// Iterations metadata.
		if len(resp.Usage.Iterations) > 0 {
			if providerMeta == nil {
				providerMeta = map[string]any{}
			}
			iters := make([]map[string]any, len(resp.Usage.Iterations))
			for i, iter := range resp.Usage.Iterations {
				iters[i] = map[string]any{
					"type":         iter.Type,
					"inputTokens":  iter.InputTokens,
					"outputTokens": iter.OutputTokens,
				}
			}
			providerMeta["iterations"] = iters
		}
	}

	// Context management metadata.
	if resp.ContextManagement != nil && len(resp.ContextManagement.AppliedEdits) > 0 {
		if providerMeta == nil {
			providerMeta = map[string]any{}
		}
		providerMeta["contextManagement"] = map[string]any{
			"appliedEdits": resp.ContextManagement.AppliedEdits,
		}
	}

	// Container metadata.
	if resp.Container != nil {
		if providerMeta == nil {
			providerMeta = map[string]any{}
		}
		container := map[string]any{
			"expiresAt": resp.Container.ExpiresAt,
			"id":        resp.Container.ID,
		}
		if len(resp.Container.Skills) > 0 {
			skills := make([]map[string]any, len(resp.Container.Skills))
			for i, s := range resp.Container.Skills {
				skills[i] = map[string]any{
					"type":    s.Type,
					"skillId": s.SkillID,
					"version": s.Version,
				}
			}
			container["skills"] = skills
		}
		providerMeta["container"] = container
	}

	// Attach provider metadata to response.
	if providerMeta != nil {
		result.Response.ProviderMetadata = providerMeta
	}

	return result, nil
}

// --- HTTP helpers ---

func (m *chatModel) doHTTP(ctx context.Context, body map[string]any, toolBetas ...string) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Extract per-request headers before marshaling.
	reqHeaders, _ := body["_headers"].(map[string]string)
	delete(body, "_headers")

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+"/v1/messages", jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", token)
	req.Header.Set("anthropic-version", apiVersion)
	// Merge base betas with tool-specific betas.
	allBetas := betaFeatures
	if len(toolBetas) > 0 {
		seen := make(map[string]bool)
		for b := range strings.SplitSeq(betaFeatures, ",") {
			seen[b] = true
		}
		for _, b := range toolBetas {
			if !seen[b] {
				allBetas += "," + b
				seen[b] = true
			}
		}
	}
	req.Header.Set("anthropic-beta", allBetas)

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	for k, v := range reqHeaders {
		req.Header.Set(k, v)
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("anthropic", resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

func (m *chatModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}

func (m *chatModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}
