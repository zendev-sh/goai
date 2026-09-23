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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel          = (*chatModel)(nil)
	_ provider.CapableModel           = (*chatModel)(nil)
	_ provider.FileUploadCapableModel = (*chatModel)(nil)
)

const (
	defaultBaseURL = "https://api.anthropic.com"
	apiVersion     = "2023-06-01"
	// betaFeatures is the baseline anthropic-beta header sent on every request.
	// It intentionally excludes claude-code-20250219 (only needed when a
	// feature that depends on it is used, e.g. the container option) so plain
	// Messages requests are not gated unnecessarily.
	betaFeatures     = "interleaved-thinking-2025-05-14"
	defaultMaxTokens = 16384

	// Feature-gated beta headers, applied only when the corresponding field is
	// present on the request.
	betaContextManagement = "context-1m-2025-08-07"
	betaFastMode          = "fast-mode-2026-02-01"
	betaClaudeCode        = "claude-code-20250219"
	betaThinkingBinding   = "thinking-binding-controls-2026-08-01"
)

// anthropicHandledKeys lists provider option keys that are explicitly handled
// in buildRequest and must not be passed through verbatim.
// Allocated once at package init to avoid per-request map allocation.
var anthropicHandledKeys = map[string]bool{
	"thinking": true, "_headers": true, "output_format": true,
	"disableParallelToolUse": true, "effort": true, "speed": true,
	"container": true, "contextManagement": true,
}

// anthropicProtectedKeys lists wire-format keys that must not be overwritten
// by provider option passthrough.
// Allocated once at package init to avoid per-request map allocation.
var anthropicProtectedKeys = map[string]bool{
	"model": true, "stream": true, "messages": true,
	"max_tokens": true, "system": true, "temperature": true,
	"top_p": true, "top_k": true, "stop_sequences": true,
	"tools": true, "tool_choice": true,
	// SDK-internal keys that are never sent on the wire.
	"structuredOutputMode": true, "sendReasoning": true,
	"cacheControl": true, "streamingTransport": true,
}

// Option configures the Anthropic provider.
type Option func(*options)

// AuthMode controls how the auth token is sent in HTTP requests.
type AuthMode int

const (
	// AuthAPIKey sends the token as x-api-key header (default Anthropic).
	AuthAPIKey AuthMode = iota
	// AuthBearer sends the token as Authorization: Bearer header (for Vertex AI).
	AuthBearer
)

// URLBuilder constructs the request URL from the base URL, model ID, and streaming flag.
type URLBuilder func(baseURL, modelID string, streaming bool) string

// BodyTransformer modifies the request body before it is sent.
type BodyTransformer func(body map[string]any) map[string]any

type options struct {
	tokenSource     provider.TokenSource
	baseURL         string
	headers         map[string]string
	httpClient      *http.Client
	authMode        AuthMode
	urlBuilder      URLBuilder
	bodyTransformer BodyTransformer
	errorProvider   string // provider name for error parsing (default "anthropic")
	skipEnvResolve  bool   // skip ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL env resolution

	// nativeOutputFormatModels restricts the model IDs for which native
	// structured output (output_config.format) is enabled in "auto" mode.
	// nil selects the direct-Anthropic documented list; an empty slice
	// disables native structured output entirely. Platform adapters set this
	// to their own documented compatibility set.
	nativeOutputFormatModels []string

	// autoStreaming enables DoGenerate to transparently issue stream:true and
	// reassemble the response for long-running (thinking) requests. Enabled by
	// default for the direct API; adapters whose streaming endpoint needs
	// extra permissions (Bedrock's InvokeModelWithResponseStream) opt out.
	autoStreaming bool
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

// WithAuthMode sets how the auth token is sent (API key header vs Bearer token).
func WithAuthMode(mode AuthMode) Option {
	return func(o *options) {
		o.authMode = mode
	}
}

// WithURLBuilder overrides URL construction for each request.
// The function receives the base URL, model ID, and whether this is a streaming request.
func WithURLBuilder(fn URLBuilder) Option {
	return func(o *options) {
		o.urlBuilder = fn
	}
}

// WithBodyTransformer sets a function to modify the request body before sending.
// Used by Vertex Anthropic to remove "model" and add "anthropic_version".
func WithBodyTransformer(fn BodyTransformer) Option {
	return func(o *options) {
		o.bodyTransformer = fn
	}
}

// WithErrorProvider overrides the provider name used in error parsing (default "anthropic").
func WithErrorProvider(name string) Option {
	return func(o *options) {
		o.errorProvider = name
	}
}

// WithSkipEnvResolve disables automatic resolution of ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL
// environment variables. Used by Vertex Anthropic which has its own auth and URL resolution.
func WithSkipEnvResolve() Option {
	return func(o *options) {
		o.skipEnvResolve = true
	}
}

// NativeOutputFormatSupport selects which documented model set enables native
// structured output (output_config.format) in "auto" mode for a platform.
type NativeOutputFormatSupport int

const (
	// NativeOutputFormatDirect is the default: the full documented Claude API
	// compatibility set. Used by the direct API and platforms (Vertex, Azure)
	// that expose the same set.
	NativeOutputFormatDirect NativeOutputFormatSupport = iota
	// NativeOutputFormatBedrock is the narrower documented Amazon Bedrock set.
	NativeOutputFormatBedrock
	// NativeOutputFormatDisabled disables native structured output entirely;
	// the platform does not document support for the field.
	NativeOutputFormatDisabled
)

// WithNativeOutputFormatSupport restricts which models enable native
// structured output (output_config.format) in "auto" mode. Platform adapters
// whose documented compatibility differs from the direct Claude API set this
// explicitly (e.g. bedrock.WithNativeOutputFormatSupport(Bedrock),
// minimax.WithNativeOutputFormatSupport(Disabled)).
func WithNativeOutputFormatSupport(support NativeOutputFormatSupport) Option {
	return func(o *options) {
		switch support {
		case NativeOutputFormatBedrock:
			o.nativeOutputFormatModels = bedrockNativeOutputFormatModels
		case NativeOutputFormatDisabled:
			o.nativeOutputFormatModels = []string{}
		default:
			o.nativeOutputFormatModels = nil
		}
	}
}

// WithAutoStreaming controls whether DoGenerate transparently issues stream:true
// and reassembles the response for long-running (thinking) requests. Enabled by
// default for the direct API. Adapters whose streaming endpoint needs extra
// permissions (Bedrock's InvokeModelWithResponseStream) disable it so existing
// deployments do not regress by default.
func WithAutoStreaming(enabled bool) Option {
	return func(o *options) {
		o.autoStreaming = enabled
	}
}

// Chat creates an Anthropic language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL, autoStreaming: true}
	for _, opt := range opts {
		opt(&o)
	}
	if !o.skipEnvResolve {
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

// anthropicModelVersionPattern matches the generation numbers in a
// current-naming Anthropic model id ("claude-<family>-<major>[-<minor>]").
//
// Deliberately unanchored: Bedrock reuses this provider via
// bedrock.AnthropicChat with a prefixed id ("anthropic.claude-opus-5",
// "us.anthropic.claude-sonnet-4-6"), and Vertex appends an "@date" suffix
// ("claude-opus-4-5@20251101"). Legacy family-last ids ("claude-3-7-sonnet",
// "claude-3-5-sonnet-20241022") do not match and are reported as unversioned.
var anthropicModelVersionPattern = regexp.MustCompile(`claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?`)

// anthropicModelVersion extracts the major and minor generation numbers from a
// current-naming Anthropic model id. ok is false when the id carries no
// parseable version, in which case callers should fall back to legacy handling.
//
// A trailing numeric segment is only treated as a minor version when it is one
// or two digits; longer runs are release dates, not versions, so
// "claude-sonnet-4-20250514" reports major 4 with no minor rather than minor
// 20250514.
func anthropicModelVersion(modelID string) (major, minor int, ok bool) {
	m := anthropicModelVersionPattern.FindStringSubmatch(strings.ToLower(modelID))
	if m == nil {
		return 0, 0, false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, 0, false
	}
	if len(m[2]) > 0 && len(m[2]) <= 2 {
		// Cannot fail: the pattern matched one or two ASCII digits.
		minor, _ = strconv.Atoi(m[2])
	}
	return major, minor, true
}

// supportsThinking returns true for Anthropic models that support extended
// thinking: every model from the Claude 4 generation onward, plus the legacy
// claude-3-7-sonnet.
//
// Version-derived rather than a literal model list, which had gone stale for
// the 5.x generation (claude-opus-5, claude-sonnet-5, claude-fable-5) and for
// claude-haiku-4-5, all of which support thinking but matched none of the
// previous substrings.
func supportsThinking(modelID string) bool {
	// Legacy and non-versioned aliases that support thinking but carry no
	// parseable generation number.
	if strings.Contains(modelID, "claude-3-7-sonnet") || strings.Contains(modelID, "claude-mythos-preview") {
		return true
	}
	major, _, ok := anthropicModelVersion(modelID)
	return ok && major >= 4
}

// hasRemoteRef returns true if any message part contains a RemoteRef.
func hasRemoteRef(msgs []provider.Message) bool {
	for _, msg := range msgs {
		for _, part := range msg.Content {
			if part.RemoteRef != nil {
				return true
			}
		}
	}
	return false
}

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: true,
		Reasoning:   supportsThinking(m.id),
		ToolCall:    true,
		Attachment:  true,
		FileUpload:  true,
		InputModalities: provider.ModalitySet{
			Text:  true,
			Image: true,
			PDF:   true,
		},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) FileUploader() provider.FileUploader {
	return &fileUploader{opts: m.opts}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	useOutputFormat := m.useNativeOutputFormat(params)
	rfMode := params.ResponseFormat != nil && !useOutputFormat
	if rfMode {
		params = injectResponseFormatTool(params)
	} else if useOutputFormat {
		var err error
		params, err = injectNativeOutputFormat(params)
		if err != nil {
			return nil, err
		}
	}
	// Long-running requests are issued with stream:true and reassembled
	// into the same Message document a non-streaming call would have
	// returned. Transport only — the result is indistinguishable to the
	// caller. See useStreamingTransport.
	streaming, err := m.useStreamingTransport(params)
	if err != nil {
		return nil, err
	}
	body := m.buildRequest(params, streaming)
	toolBetas := collectToolBetas(params.Tools)
	toolBetas = append(toolBetas, collectRequestBetas(body)...)
	if hasRemoteRef(params.Messages) {
		toolBetas = append(toolBetas, filesBetaHeader)
	}

	resp, err := m.doHTTP(ctx, body, toolBetas...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var respBody []byte
	if streaming {
		respBody, err = accumulateStreamedMessage(ctx, resp.Body)
	} else {
		// Cap the non-streaming response body at 64 MiB so a runaway/hostile
		// response cannot exhaust memory.
		const maxNonStreamResponseBytes = 64 << 20 // 64 MiB
		respBody, err = io.ReadAll(io.LimitReader(resp.Body, maxNonStreamResponseBytes+1))
		if err != nil {
			return nil, fmt.Errorf("reading response: %w", err)
		}
		if len(respBody) > maxNonStreamResponseBytes {
			return nil, fmt.Errorf("anthropic: response body exceeds %d bytes", maxNonStreamResponseBytes)
		}
	}
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
		var err error
		params, err = injectNativeOutputFormat(params)
		if err != nil {
			return nil, err
		}
	}
	body := m.buildRequest(params, true)
	toolBetas := collectToolBetas(params.Tools)
	toolBetas = append(toolBetas, collectRequestBetas(body)...)
	if hasRemoteRef(params.Messages) {
		toolBetas = append(toolBetas, filesBetaHeader)
	}

	resp, err := m.doHTTP(ctx, body, toolBetas...)
	if err != nil {
		return nil, err
	}
	// Keep resp.Request and its serialized request body out of the stream closure.
	responseBody := resp.Body

	return provider.RunStream(ctx, responseBody, func(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
		parseSSE(ctx, body, out, rfMode)
	}), nil
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
			systemPart["cache_control"] = ephemeralCacheControl(params.CacheTTL)
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
			// Anthropic supports an explicit "none" tool_choice type that
			// prevents tool use while keeping tools registered. Send it
			// instead of deleting the tools array.
			body["tool_choice"] = map[string]any{"type": "none"}
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
	// Adaptive thinking accepts an optional display: "summarized" | "omitted"
	// (Opus 4.7/4.8 default to "omitted", so thinking text streams empty unless
	// "summarized" is requested).
	if thinking, ok := params.ProviderOptions["thinking"]; ok {
		if tm, ok := thinking.(map[string]any); ok {
			thinkingReq := map[string]any{}
			if t, ok := tm["type"]; ok {
				thinkingReq["type"] = t
			}
			if budget, ok := tm["budgetTokens"]; ok {
				thinkingReq["budget_tokens"] = budget
			}
			if display, ok := tm["display"]; ok {
				thinkingReq["display"] = display
			}
			// Binding-only recovery requests need not specify a thinking type.
			// The policy is always explicit; never opt callers into drop_block.
			if binding, ok := tm["blockBinding"].(map[string]any); ok {
				if behavior, ok := binding["prefixMismatchBehavior"]; ok {
					thinkingReq["block_binding"] = map[string]any{"prefix_mismatch_behavior": behavior}
				}
			} else if binding, ok := tm["block_binding"].(map[string]any); ok {
				thinkingReq["block_binding"] = maps.Clone(binding)
			}
			if len(thinkingReq) > 0 {
				body["thinking"] = thinkingReq
			}

			// Extended thinking is incompatible with a forced tool_choice
			// ("required" / a specific tool). The API rejects the combination,
			// so downgrade to "auto" so the model may still choose to call a
			// tool while thinking.
			if t, _ := tm["type"].(string); t == "enabled" {
				if tc, ok := body["tool_choice"].(map[string]any); ok {
					switch tc["type"] {
					case "any", "tool":
						body["tool_choice"] = map[string]any{"type": "auto"}
					}
				}
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
					"type":                      "auto",
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

	// output_format -- native structured-output schema set by
	// injectNativeOutputFormat. Anthropic nests it under output_config.format
	// (the top-level output_format field is deprecated). Merge into
	// output_config so it coexists with effort.
	// https://platform.claude.com/docs/en/build-with-claude/structured-outputs
	if of, ok := params.ProviderOptions["output_format"]; ok {
		if ofm, ok := of.(map[string]any); ok {
			oc, _ := body["output_config"].(map[string]any)
			if oc == nil {
				oc = map[string]any{}
			}
			oc["format"] = ofm
			body["output_config"] = oc
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
	for k, v := range params.ProviderOptions {
		if anthropicHandledKeys[k] || anthropicProtectedKeys[k] {
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
	// Pre-process: fix orphaned tool-calls, merge consecutive roles, reorder parts.
	msgs = provider.NormalizeToolMessages(msgs)
	msgs = provider.ReorderAssistantParts(msgs)

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
				applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartReasoning:
				// Signature is required for replaying thinking blocks, including
				// omitted-display blocks whose thinking text is intentionally empty.
				var sig string
				if part.ProviderOptions != nil {
					sig, _ = part.ProviderOptions["signature"].(string)
				}
				if sig != "" {
					p := map[string]any{"type": "thinking", "thinking": part.Text, "signature": sig}
					applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
					content = append(content, p)
				} else if part.ProviderOptions != nil {
					// Redacted thinking (no text, just encrypted data).
					if data, ok := part.ProviderOptions["redactedData"].(string); ok && data != "" {
						p := map[string]any{"type": "redacted_thinking", "data": data}
						content = append(content, p)
					}
				}

			case provider.PartImage:
				if part.RemoteRef != nil {
					// A remotely uploaded image is referenced by file ID.
					p := map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":    "file",
							"file_id": part.RemoteRef.ID,
						},
					}
					applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
					content = append(content, p)
					break
				}
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
				applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
				content = append(content, p)

			case provider.PartFile:
				if part.RemoteRef != nil {
					p := map[string]any{
						"type": "document",
						"source": map[string]any{
							"type":    "file",
							"file_id": part.RemoteRef.ID,
						},
					}
					applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
					content = append(content, p)
				} else if part.URL != "" {
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
					applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
					content = append(content, p)
				}

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
				// Server-executed tools (e.g. web_search) carry the matched
				// result block via ProviderOptions["resultBlock"]. Emit
				// "server_tool_use" instead of "tool_use" so the API
				// recognizes the pairing, then append the raw result block.
				resultBlock, _ := part.ProviderOptions["resultBlock"].(map[string]any)
				toolUseType := "tool_use"
				if resultBlock != nil {
					toolUseType = "server_tool_use"
				}
				p := map[string]any{
					"type":  toolUseType,
					"id":    part.ToolCallID,
					"name":  part.ToolName,
					"input": input,
				}
				if resultBlock == nil {
					applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
				}
				content = append(content, p)
				if resultBlock != nil {
					applyCacheControl(resultBlock, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
					content = append(content, resultBlock)
				}

			case provider.PartToolResult:
				p := map[string]any{
					"type":        "tool_result",
					"tool_use_id": part.ToolCallID,
					"content":     part.ToolOutput,
				}
				applyCacheControl(p, part.CacheControl, part.CacheControlTTL, msgCacheControl, isLast)
				content = append(content, p)
			}
		}

		// Anthropic rejects messages with empty content arrays.
		if len(content) == 0 {
			continue
		}

		m["content"] = content

		// Alternating user/assistant - safety net merge (NormalizeToolMessages
		// already merged at provider.Message level).
		if len(result) > 0 && result[len(result)-1]["role"] == role {
			existing, ok := result[len(result)-1]["content"].([]map[string]any)
			if ok {
				result[len(result)-1]["content"] = append(existing, content...)
				continue
			}
		}

		result = append(result, m)
	}
	return result
}

// ephemeralCacheControl builds an ephemeral cache_control marker, attaching the
// TTL ("5m"/"1h") when non-empty. Empty TTL preserves the Anthropic default (5m).
// The 1h TTL needs no beta header (GA).
func ephemeralCacheControl(ttl string) map[string]any {
	cc := map[string]any{"type": "ephemeral"}
	if ttl != "" {
		cc["ttl"] = ttl
	}
	return cc
}

// applyCacheControl adds cache_control to a content part.
// Part-level CacheControl takes precedence; message-level only applies to the last part.
// partTTL ("5m"/"1h") is attached to the part-level marker; empty = provider default (5m).
func applyCacheControl(p map[string]any, partCC, partTTL string, msgCC map[string]any, isLast bool) {
	if partCC != "" {
		cc := map[string]any{"type": partCC}
		if partTTL != "" {
			cc["ttl"] = partTTL
		}
		p["cache_control"] = cc
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

// directNativeOutputFormatModels is the documented Claude API compatibility set
// for native structured output (output_config.format), kept explicit so a
// future model name is not enabled before the platform documentation
// guarantees it. Release-date aliases (e.g. claude-opus-4-5-20251101) match via
// modelIDMatches.
//
// Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#compatibility
var directNativeOutputFormatModels = []string{
	"claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
	"claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
	"claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5",
	"claude-opus-4-5", "claude-haiku-4-5",
}

// bedrockNativeOutputFormatModels is the documented Amazon Bedrock subset.
// Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#compatibility
var bedrockNativeOutputFormatModels = []string{
	"claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5",
	"claude-opus-4-5", "claude-haiku-4-5",
}

// supportsNativeOutputFormat reports whether the model supports native
// structured output. It consults the adapter-provided compatibility list when
// set (Bedrock, Vertex, Azure, MiniMax), otherwise the direct-Anthropic list.
func (m *chatModel) supportsNativeOutputFormat() bool {
	models := m.opts.nativeOutputFormatModels
	if models == nil {
		models = directNativeOutputFormatModels
	}
	return modelMatchesAny(m.id, models)
}

// modelMatchesAny reports whether modelID matches any base model name in
// models, allowing documented release-date, @date, and -v suffixed aliases but
// rejecting unknown future numeric families.
func modelMatchesAny(modelID string, models []string) bool {
	id := strings.ToLower(modelID)
	for _, base := range models {
		if modelIDMatches(id, base) {
			return true
		}
	}
	return false
}

func modelIDMatches(id, base string) bool {
	idx := strings.Index(id, base)
	if idx < 0 {
		return false
	}
	// The base must be a whole token: preceded by a separator ('.' for Bedrock
	// prefixes like "anthropic.claude-opus-5") or the start of the string.
	if idx > 0 {
		prev := id[idx-1]
		if prev != '.' && prev != '-' && prev != '/' && prev != ':' {
			return false
		}
	}
	suffix := id[idx+len(base):]
	if suffix == "" {
		return true
	}
	// Vertex @date suffix or Bedrock -v versioned suffix.
	if strings.HasPrefix(suffix, "@") || strings.HasPrefix(suffix, "-v") {
		return true
	}
	// Release-date alias: claude-opus-4-5-20251101.
	if len(suffix) == 9 && suffix[0] == '-' {
		for i := 1; i < len(suffix); i++ {
			if suffix[i] < '0' || suffix[i] > '9' {
				return false
			}
		}
		return true
	}
	return false
}

// useStreamingTransport reports whether DoGenerate should issue its request
// with stream:true and reassemble the streamed events into a complete
// Message, instead of waiting on a single non-streaming response.
//
// Anthropic's guidance is to stream long-running requests. A non-streaming
// call holds an HTTP response open with no bytes flowing until the model has
// finished everything it intends to do, so a request that thinks for minutes
// is exposed to idle-timeout enforcement — by the client, and by any proxy
// or load balancer in between. Thinking is where that bites, and since the
// 5.x generation thinks with no thinking parameter sent at all, it is the
// common case rather than an opt-in one.
//
// This matters most in a tool loop. GenerateObject is the only structured-
// output entry point that runs tools (StreamObject is single-step by
// design), and it drives every step through DoGenerate — so before this,
// native structured output plus tools plus a thinking model was reachable
// only over the non-streaming transport, which is exactly the combination
// the guidance warns against.
//
// Callers can force the decision either way with
// ProviderOptions["streamingTransport"] = true / false. A non-boolean value
// is rejected rather than silently ignored.
func (m *chatModel) useStreamingTransport(params provider.GenerateParams) (bool, error) {
	if v, ok := params.ProviderOptions["streamingTransport"]; ok {
		b, isBool := v.(bool)
		if !isBool {
			return false, fmt.Errorf("anthropic: streamingTransport must be a boolean, got %T", v)
		}
		return b, nil
	}
	if !m.opts.autoStreaming {
		return false, nil
	}
	return m.willThink(params), nil
}

// willThink reports whether a request is expected to produce thinking, and
// therefore to run long.
//
// Two ways that happens: the caller asked for it (a "thinking" or "effort"
// provider option), or the model thinks by default with no thinking
// parameter sent — true from the 5.x generation onward. Opus 4.7/4.8 support
// adaptive thinking but only when it is requested, so they qualify via the
// first branch only.
func (m *chatModel) willThink(params provider.GenerateParams) bool {
	if !supportsThinking(m.id) {
		return false
	}
	if _, ok := params.ProviderOptions["thinking"]; ok {
		return true
	}
	if _, ok := params.ProviderOptions["effort"]; ok {
		return true
	}
	major, _, ok := anthropicModelVersion(m.id)
	return ok && major >= 5
}

// injectNativeOutputFormat stores the structured-output schema in
// ProviderOptions["output_format"]; buildRequest nests it under
// output_config.format on the wire. It returns an error when the response
// format schema is invalid or cannot be expressed for native structured
// output, so the caller surfaces it rather than silently dropping the
// requested output mode.
func injectNativeOutputFormat(params provider.GenerateParams) (provider.GenerateParams, error) {
	p := params
	// Copy the map to avoid mutating the caller's ProviderOptions.
	newOpts := maps.Clone(p.ProviderOptions)
	if newOpts == nil {
		newOpts = make(map[string]any, 1)
	}
	p.ProviderOptions = newOpts
	if p.ResponseFormat == nil {
		return params, nil
	}
	var schema any
	if len(p.ResponseFormat.Schema) > 0 {
		if err := json.Unmarshal(p.ResponseFormat.Schema, &schema); err != nil {
			return params, fmt.Errorf("anthropic: invalid response format schema: %w", err)
		}
		var err error
		schema, err = transformNativeOutputSchema(schema)
		if err != nil {
			return params, fmt.Errorf("anthropic: invalid response format schema: %w", err)
		}
	}
	p.ProviderOptions["output_format"] = map[string]any{
		"type":   "json_schema",
		"schema": schema,
	}
	// Clear ResponseFormat so the tool trick is not also applied.
	p.ResponseFormat = nil
	return p, nil
}

// supportedNativeFormats lists the string formats Anthropic's native
// structured-output validator accepts; any other format value is filtered out
// (and recorded in the description).
var supportedNativeFormats = map[string]bool{
	"date-time": true, "time": true, "date": true, "duration": true,
	"email": true, "hostname": true, "ipv4": true, "ipv6": true,
	"uuid": true, "uri": true,
}

// transformNativeOutputSchema mirrors Anthropic's documented SDK schema
// transformation for native structured output: keep only the keywords the
// validator supports for the schema's effective type, preserve unsupported
// constraints in the description, and recurse only into actual subschemas.
// Literal values (enum members, const, default) are cloned verbatim, never
// walked as schemas. The input is never mutated.
//
// Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works
func transformNativeOutputSchema(obj any) (any, error) {
	m, ok := obj.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("schema must be an object")
	}
	remaining := maps.Clone(m)
	out := make(map[string]any, len(m)+1)

	if defs, ok := remaining["$defs"]; ok {
		transformed, err := transformSchemaMap(defs)
		if err != nil {
			return nil, fmt.Errorf("$defs: %w", err)
		}
		out["$defs"] = transformed
		delete(remaining, "$defs")
	}
	if ref, ok := remaining["$ref"]; ok {
		out["$ref"] = cloneJSONValue(ref)
		return out, nil
	}

	typeName, _ := remaining["type"].(string)
	delete(remaining, "type")
	var composition string
	for _, keyword := range []string{"anyOf", "oneOf", "allOf"} {
		if _, ok := remaining[keyword].([]any); ok {
			composition = keyword
			break
		}
	}
	if composition != "" {
		transformed, err := transformSchemaList(remaining[composition])
		if err != nil {
			return nil, fmt.Errorf("%s: %w", composition, err)
		}
		if composition == "oneOf" {
			// Anthropic does not support oneOf; express it as anyOf.
			out["anyOf"] = transformed
		} else {
			out[composition] = transformed
		}
		delete(remaining, composition)
	} else if typeName != "" {
		out["type"] = typeName
	} else {
		return nil, fmt.Errorf("schema must have type, anyOf, oneOf, or allOf")
	}

	// Literal/annotation keywords are carried verbatim (never recursed).
	for _, keyword := range []string{"enum", "const", "description", "title"} {
		if value, ok := remaining[keyword]; ok {
			out[keyword] = cloneJSONValue(value)
			delete(remaining, keyword)
		}
	}

	switch typeName {
	case "object":
		if properties, ok := remaining["properties"]; ok {
			transformed, err := transformSchemaMap(properties)
			if err != nil {
				return nil, fmt.Errorf("properties: %w", err)
			}
			out["properties"] = transformed
		} else {
			out["properties"] = map[string]any{}
		}
		delete(remaining, "properties")
		// Anthropic requires additionalProperties:false on objects.
		delete(remaining, "additionalProperties")
		out["additionalProperties"] = false
		if required, ok := remaining["required"]; ok {
			out["required"] = cloneJSONValue(required)
			delete(remaining, "required")
		}
	case "array":
		if items, ok := remaining["items"]; ok {
			transformed, err := transformNativeOutputSchema(items)
			if err != nil {
				return nil, fmt.Errorf("items: %w", err)
			}
			out["items"] = transformed
			delete(remaining, "items")
		}
		// Anthropic accepts minItems only as 0 or 1; anything else is
		// unsupported and falls through to the description.
		if minItems, ok := remaining["minItems"].(float64); ok && (minItems == 0 || minItems == 1) {
			out["minItems"] = minItems
			delete(remaining, "minItems")
		}
	case "string":
		if format, ok := remaining["format"].(string); ok && supportedNativeFormats[format] {
			out["format"] = format
			delete(remaining, "format")
		}
	case "integer", "number", "boolean", "null", "":
	default:
		return nil, fmt.Errorf("unsupported schema type %q", typeName)
	}

	// Anything left (pattern, patternProperties, dependentSchemas, unsupported
	// constraints, etc.) is not expressible as a native constraint; record it
	// in the description so it is not silently dropped.
	if len(remaining) > 0 {
		extra := formatUnsupportedSchemaKeywords(remaining)
		if description, _ := out["description"].(string); description != "" {
			out["description"] = description + "\n\n" + extra
		} else {
			out["description"] = extra
		}
	}
	return out, nil
}

func transformSchemaMap(value any) (any, error) {
	m, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("must be an object")
	}
	out := make(map[string]any, len(m))
	for name, schema := range m {
		transformed, err := transformNativeOutputSchema(schema)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		out[name] = transformed
	}
	return out, nil
}

func transformSchemaList(value any) (any, error) {
	items, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("must be an array")
	}
	out := make([]any, len(items))
	for i, schema := range items {
		transformed, err := transformNativeOutputSchema(schema)
		if err != nil {
			return nil, fmt.Errorf("item %d: %w", i, err)
		}
		out[i] = transformed
	}
	return out, nil
}

func formatUnsupportedSchemaKeywords(values map[string]any) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s: %v", key, values[key]))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

func cloneJSONValue(value any) any {
	switch v := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for k, item := range v {
			out[k] = cloneJSONValue(item)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = cloneJSONValue(item)
		}
		return out
	default:
		return value
	}
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

// inputTransformations normalizes the generic SSE JSON value to the same Go
// type as parseResponse. Null/absent remains nil; an empty array stays non-nil.
func inputTransformations(value any) ([]map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	entries, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("anthropic: input_transformations must be an array")
	}
	result := make([]map[string]any, len(entries))
	for i, entry := range entries {
		if entry == nil {
			continue
		}
		block, ok := entry.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("anthropic: input_transformations[%d] must be an object", i)
		}
		result[i] = block
	}
	return result, nil
}

func parseSSE(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk, isRFMode bool) {
	defer close(out)

	sseScanner := sse.NewScanner(body)

	var currentToolCallID string
	var currentToolName string
	var currentToolArgs strings.Builder // accumulate partial JSON fragments
	var isRFBlock bool                  // true when current tool_use block is the synthetic response format tool
	var isFirstDelta bool               // true for first input_json_delta of a tool_use block
	var isServerTool bool               // true when current block is server_tool_use
	var isResultBlock bool              // true when current block is a server tool result (e.g. web_search_tool_result)
	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	var finishMeta map[string]any // metadata accumulated for ChunkFinish
	finish := func(reason provider.FinishReason) {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
		meta := maps.Clone(finishMeta)
		if meta != nil {
			responseMeta.ProviderMetadata = maps.Clone(finishMeta)
			meta["providerMetadata"] = map[string]map[string]any{"anthropic": maps.Clone(finishMeta)}
		}
		provider.TrySend(ctx, out, provider.StreamChunk{
			Type: provider.ChunkFinish, FinishReason: reason, Usage: usage,
			Response: responseMeta, Metadata: meta,
		})
	}

	// Pending server_tool_use ChunkToolCalls keyed by tool_use_id, deferred so
	// each can be paired with its own result block before emitting. Keyed by ID
	// (not a single slot) so parallel server tools each keep their result.
	// See parseResponse for the non-streaming equivalent.
	type pendingServerCall struct {
		name        string
		args        string
		resultBlock map[string]any
	}
	pendingCalls := make(map[string]*pendingServerCall)
	currentResultUseID := ""

	flushPendingCall := func(id string) bool {
		pc, ok := pendingCalls[id]
		if !ok {
			return true
		}
		args := cmp.Or(pc.args, "{}")
		chunk := provider.StreamChunk{
			Type:       provider.ChunkToolCall,
			ToolCallID: id,
			ToolName:   pc.name,
			ToolInput:  args,
		}
		if pc.resultBlock != nil {
			chunk.Metadata = map[string]any{"resultBlock": pc.resultBlock}
		}
		delete(pendingCalls, id)
		return provider.TrySend(ctx, out, chunk)
	}

	flushAllPending := func() bool {
		for id := range pendingCalls {
			if !flushPendingCall(id) {
				return false
			}
		}
		return true
	}

	for data, ok := sseScanner.Next(); ok; data, ok = sseScanner.Next() {

		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		eventType, _ := event["type"].(string)

		switch eventType {
		case "message_start":
			if msg, ok := event["message"].(map[string]any); ok {
				transformations, err := inputTransformations(msg["input_transformations"])
				if err != nil {
					provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: err})
					return
				}
				if transformations != nil {
					if finishMeta == nil {
						finishMeta = map[string]any{}
					}
					finishMeta["inputTransformations"] = transformations
				}
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
				isResultBlock = false
				idx, _ := streamEventIndex(event)
				blockID := strconv.Itoa(idx)
				// If pending server_tool_use calls await their result blocks and
				// the next block is neither a result nor another server tool,
				// their results are not coming this step: flush without
				// resultBlock attached.
				if len(pendingCalls) > 0 && !isServerToolResultBlock(cbType) && cbType != "server_tool_use" {
					if !flushAllPending() {
						return
					}
				}
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
				case "redacted_thinking":
					// Redacted thinking arrives as a complete block in
					// content_block_start (no deltas). Surface the encrypted
					// data so it can be replayed on the next turn.
					if data, _ := cb["data"].(string); data != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type: provider.ChunkReasoning,
							Text: "",
							Metadata: map[string]any{
								"redactedData": data,
								"blockId":      blockID,
							},
						}) {
							return
						}
					}
				default:
					if isServerToolResultBlock(cbType) {
						isResultBlock = true
						// Anthropic emits the full result block in
						// content_block_start (no input_json_delta for
						// these). Capture verbatim for round-trip when
						// tool_use_id matches a pending call.
						currentResultUseID, _ = cb["tool_use_id"].(string)
						if pc, ok := pendingCalls[currentResultUseID]; ok {
							pc.resultBlock = cb
						}
					}
				}
			}

		case "content_block_delta":
			if delta, ok := event["delta"].(map[string]any); ok {
				deltaType, _ := delta["type"].(string)
				idx, idxErr := streamEventIndex(event)
				if idxErr != nil {
					idx = -1 // names no real block
				}
				blockID := strconv.Itoa(idx)
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
						if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: text, Metadata: map[string]any{"blockId": blockID}}) {
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
								"blockId":   blockID,
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
			switch {
			case isResultBlock:
				// End of a server tool result block. Flush the deferred
				// server_tool_use ChunkToolCall with resultBlock attached.
				if !flushPendingCall(currentResultUseID) {
					return
				}
				isResultBlock = false
				currentResultUseID = ""
			case isServerTool && currentToolCallID != "":
				// Defer ChunkToolCall emission so we can attach the matching
				// result block (next content_block) before flushing.
				pendingCalls[currentToolCallID] = &pendingServerCall{
					name: currentToolName,
					args: currentToolArgs.String(),
				}
				currentToolArgs.Reset()
			case currentToolCallID != "" && !isRFBlock:
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
			transformations, err := inputTransformations(event["input_transformations"])
			if err != nil {
				provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: err})
				return
			}
			if transformations != nil {
				if finishMeta == nil {
					finishMeta = map[string]any{}
				}
				finishMeta["inputTransformations"] = transformations
			}
			if delta, ok := event["delta"].(map[string]any); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					// Flush any pending server tool calls before signalling
					// step finish so consumers see the ChunkToolCall first.
					if !flushAllPending() {
						return
					}
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
			// Flush pending server tool calls without resultBlock if none arrived.
			if !flushAllPending() {
				return
			}
			finish("")
			return

		case "error":
			handleStreamError(ctx, data, event, out)
			return
		}
	}

	if err := sseScanner.Err(); err != nil {
		if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) {
			return
		}
		finish(provider.FinishError)
		return
	}
	// Clean EOF without message_stop: emit finish with accumulated usage and response meta.
	_ = flushAllPending()
	finish("")
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
		chunk = provider.StreamChunk{Type: provider.ChunkError, Error: &goai.APIError{Message: msg, ResponseBody: data}}
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

// serverToolResultBlockTypes lists Anthropic block types that pair with a
// server_tool_use as the matched result. They must be round-tripped on the
// assistant turn -- omitting them causes the API to reject re-sent transcripts
// with "tool_use ids were found without `tool_result` blocks".
var serverToolResultBlockTypes = map[string]bool{
	"web_search_tool_result":                 true,
	"web_fetch_tool_result":                  true,
	"code_execution_tool_result":             true,
	"bash_code_execution_tool_result":        true,
	"text_editor_code_execution_tool_result": true,
	"mcp_tool_result":                        true,
	"tool_search_tool_result":                true,
}

func isServerToolResultBlock(t string) bool {
	return serverToolResultBlockTypes[t]
}

// accumulateStreamedMessage reads Anthropic's SSE event stream and
// reassembles the single Message object those events describe, returning the
// JSON bytes as though the request had been issued non-streaming.
//
// Rebuilding the wire document, rather than mapping events onto
// provider.GenerateResult directly, is the deliberate part. parseResponse is
// the one place that knows how to interpret a Message, and it reads several
// things provider.StreamChunk has no field for: thinking-block signatures
// (which must be replayed verbatim on the next turn), redacted_thinking
// payloads, server tool result blocks, citations, and container /
// context_management metadata. Reassembling the document keeps DoGenerate's
// two transports identical in everything but transport; mapping events would
// fork that logic and silently drop whatever a chunk cannot carry.
//
// An "error" event is returned as-is: parseResponse already recognises the
// {"type":"error"} envelope and converts it to an APIError or
// ContextOverflowError, so error classification stays in one place too.
// maxStreamedContentBlocks bounds the number of content blocks a reassembled
// stream may contain, so a hostile index cannot force unbounded allocation.
const maxStreamedContentBlocks = 500

func accumulateStreamedMessage(ctx context.Context, body io.Reader) ([]byte, error) {
	scanner := sse.NewScanner(body)

	var message map[string]any
	var content []map[string]any
	// Anthropic sends tool input as a JSON string split across
	// input_json_delta fragments, keyed by content block index. Fragments are
	// only valid JSON once concatenated, so they are buffered until
	// content_block_stop.
	toolInput := map[int]*strings.Builder{}
	// Lifecycle tracking: message_start must precede every other event,
	// message_stop must terminate a well-formed stream, and every started
	// content block must be stopped before the stream ends. Violations are
	// protocol errors, never partial successes.
	messageStarted := false
	messageStopped := false
	openBlocks := map[int]bool{}

	blockAt := func(idx int) map[string]any {
		for len(content) <= idx {
			content = append(content, map[string]any{})
		}
		return content[idx]
	}

	for {
		ev, ok := scanner.NextEvent()
		if !ok {
			break
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// Bedrock emits event:error with AWS exception payloads that carry no
		// Anthropic "type" field. Surface it as a protocol error rather than
		// letting it fall through to a partial success.
		if ev.Type == "error" {
			return nil, fmt.Errorf("anthropic: stream error event: %s", string(ev.Data))
		}
		var event map[string]any
		if err := json.Unmarshal(ev.Data, &event); err != nil {
			return nil, fmt.Errorf("anthropic: malformed stream event: %w", err)
		}

		switch eventType, _ := event["type"].(string); eventType {
		case "message_start":
			if messageStarted {
				return nil, fmt.Errorf("anthropic: duplicate message_start")
			}
			messageStarted = true
			msg, ok := event["message"].(map[string]any)
			if !ok {
				return nil, fmt.Errorf("anthropic: message_start missing message")
			}
			message = msg
			// content arrives as separate events; rebuild it from those rather
			// than trusting the (usually empty) array in the envelope.
			if existing, ok := msg["content"].([]any); ok {
				for _, b := range existing {
					if bm, ok := b.(map[string]any); ok {
						content = append(content, bm)
						openBlocks[len(content)-1] = true
					}
				}
			}

		case "content_block_start":
			if !messageStarted {
				return nil, fmt.Errorf("anthropic: content_block_start before message_start")
			}
			idx, err := streamEventIndex(event)
			if err != nil {
				return nil, err
			}
			if openBlocks[idx] {
				return nil, fmt.Errorf("anthropic: duplicate content_block_start at index %d", idx)
			}
			block, _ := event["content_block"].(map[string]any)
			if block == nil {
				block = map[string]any{}
			}
			blockAt(idx)
			content[idx] = block
			openBlocks[idx] = true

		case "content_block_delta":
			if !messageStarted {
				return nil, fmt.Errorf("anthropic: content_block_delta before message_start")
			}
			idx, err := streamEventIndex(event)
			if err != nil {
				return nil, err
			}
			if !openBlocks[idx] {
				return nil, fmt.Errorf("anthropic: content_block_delta for unopened block %d", idx)
			}
			delta, _ := event["delta"].(map[string]any)
			if delta == nil {
				continue
			}
			block := blockAt(idx)
			switch deltaType, _ := delta["type"].(string); deltaType {
			case "text_delta":
				appendStringField(block, "text", delta["text"])
			case "thinking_delta":
				appendStringField(block, "thinking", delta["thinking"])
			case "signature_delta":
				appendStringField(block, "signature", delta["signature"])
			case "input_json_delta":
				if fragment, ok := delta["partial_json"].(string); ok {
					sb := toolInput[idx]
					if sb == nil {
						sb = &strings.Builder{}
						toolInput[idx] = sb
					}
					sb.WriteString(fragment)
				}
			case "citations_delta":
				if citation, ok := delta["citation"].(map[string]any); ok {
					existing, _ := block["citations"].([]any)
					block["citations"] = append(existing, citation)
				}
			}

		case "content_block_stop":
			if !messageStarted {
				return nil, fmt.Errorf("anthropic: content_block_stop before message_start")
			}
			idx, err := streamEventIndex(event)
			if err != nil {
				return nil, err
			}
			if !openBlocks[idx] {
				return nil, fmt.Errorf("anthropic: content_block_stop for unopened block %d", idx)
			}
			delete(openBlocks, idx)
			sb := toolInput[idx]
			delete(toolInput, idx)
			if sb == nil {
				continue
			}
			var input any
			if err := json.Unmarshal([]byte(cmp.Or(sb.String(), "{}")), &input); err != nil {
				// Fragments that never form valid JSON are a malformed stream;
				// do not report a partial tool call as success.
				return nil, fmt.Errorf("anthropic: unparseable tool input for block %d: %w", idx, err)
			}
			blockAt(idx)["input"] = input

		case "message_delta":
			if !messageStarted {
				return nil, fmt.Errorf("anthropic: message_delta before message_start")
			}
			// stop_reason, stop_sequence and container all arrive inside
			// delta, at the same position they occupy in a complete Message.
			if delta, ok := event["delta"].(map[string]any); ok {
				for k, v := range delta {
					message[k] = v
				}
			}
			// usage and context_management arrive at the event's top level.
			if u, ok := event["usage"].(map[string]any); ok {
				existing, _ := message["usage"].(map[string]any)
				if existing == nil {
					existing = map[string]any{}
					message["usage"] = existing
				}
				// message_start's usage carries input-side counts; the delta
				// supersedes the output-side ones it repeats.
				for k, v := range u {
					existing[k] = v
				}
			}
			if cm, ok := event["context_management"]; ok {
				message["context_management"] = cm
			}
			// Like parseSSE, a null value carries no new snapshot. Only an
			// explicit empty array clears diagnostics received in message_start.
			if transformations, ok := event["input_transformations"]; ok && transformations != nil {
				message["input_transformations"] = transformations
			}

		case "message_stop":
			if !messageStarted {
				return nil, fmt.Errorf("anthropic: message_stop before message_start")
			}
			messageStopped = true

		case "error":
			// Anthropic error envelope; hand it to parseResponse unchanged so
			// error classification stays in one place.
			return ev.Data, nil
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading stream: %w", err)
	}
	if !messageStarted {
		return nil, &goai.APIError{Message: "anthropic: stream ended before message_start"}
	}
	if !messageStopped {
		return nil, fmt.Errorf("anthropic: stream ended before message_stop")
	}
	if len(openBlocks) > 0 {
		return nil, fmt.Errorf("anthropic: stream ended with %d unclosed content block(s)", len(openBlocks))
	}

	message["content"] = content
	out, err := json.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("reassembling streamed message: %w", err)
	}
	return out, nil
}

// streamEventIndex reads and validates an SSE event's content block index. A
// missing index is treated as 0. Negative, fractional, overflowing, or
// over-limit indexes are rejected as protocol errors so a hostile index cannot
// force unbounded allocation.
func streamEventIndex(event map[string]any) (int, error) {
	v, ok := event["index"]
	if !ok {
		return 0, nil
	}
	f, ok := v.(float64)
	if !ok {
		return 0, fmt.Errorf("anthropic: content block index is not a number: %T", v)
	}
	if f != float64(int(f)) || f < 0 || f > maxStreamedContentBlocks {
		return 0, fmt.Errorf("anthropic: invalid content block index %v", f)
	}
	return int(f), nil
}

// appendStringField concatenates a streamed delta onto a content block field,
// creating it when the first fragment arrives.
func appendStringField(block map[string]any, key string, value any) {
	fragment, ok := value.(string)
	if !ok {
		return
	}
	existing, _ := block[key].(string)
	block[key] = existing + fragment
}

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
			Data      string          `json:"data,omitempty"`        // redacted_thinking
			ToolUseID string          `json:"tool_use_id,omitempty"` // for server tool result blocks
			Citations []struct {
				Type            string  `json:"type"`
				CitedText       string  `json:"cited_text"`
				URL             string  `json:"url,omitempty"`
				Title           string  `json:"title,omitempty"`
				EncryptedIndex  string  `json:"encrypted_index,omitempty"`
				DocumentIndex   int     `json:"document_index,omitempty"`
				DocumentTitle   *string `json:"document_title,omitempty"`
				StartPageNumber int     `json:"start_page_number,omitempty"`
				EndPageNumber   int     `json:"end_page_number,omitempty"`
				StartCharIndex  int     `json:"start_char_index,omitempty"`
				EndCharIndex    int     `json:"end_char_index,omitempty"`
			} `json:"citations,omitempty"`
		} `json:"content"`
		StopReason           string           `json:"stop_reason"`
		InputTransformations []map[string]any `json:"input_transformations"`
		Usage                *struct {
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

	// Side-channel raw parse of content blocks: server tool result blocks
	// (e.g. web_search_tool_result) carry provider-specific payloads we cannot
	// fully model in our typed struct. We preserve them verbatim so they can
	// be round-tripped on the assistant turn (Anthropic's API rejects
	// transcripts where a server_tool_use is not immediately followed by its
	// result block).
	var rawContent struct {
		Content []json.RawMessage `json:"content"`
	}
	_ = json.Unmarshal(body, &rawContent)

	// Handle error response.
	if resp.Error != nil {
		if goai.IsOverflow(resp.Error.Message) {
			return nil, &goai.ContextOverflowError{Message: resp.Error.Message, ResponseBody: string(body)}
		}
		return nil, &goai.APIError{Message: resp.Error.Message, ResponseBody: string(body)}
	}
	if resp.Type == "error" {
		return nil, &goai.APIError{Message: "unknown error", ResponseBody: string(body)}
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
	var reasoningParts []string
	var providerMeta map[string]any
	// Map of server_tool_use ToolCall index by tool_use_id, for attaching the
	// matching result block. Keyed by ID (not a single slot) so parallel
	// server tools each get their own result block.
	serverToolIdxByID := make(map[string]int)
	for i, block := range resp.Content {
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
				reasoningParts = append(reasoningParts, block.Thinking)
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
			if block.Type == "server_tool_use" && block.ID != "" {
				serverToolIdxByID[block.ID] = len(result.ToolCalls) - 1
			}
		default:
			if isServerToolResultBlock(block.Type) && i < len(rawContent.Content) {
				idx, ok := serverToolIdxByID[block.ToolUseID]
				if !ok {
					continue
				}
				// Decode raw block as map[string]any so the request serializer
				// can re-emit it verbatim alongside the matching server_tool_use.
				var rb map[string]any
				if err := json.Unmarshal(rawContent.Content[i], &rb); err == nil {
					tc := &result.ToolCalls[idx]
					if tc.Metadata == nil {
						tc.Metadata = map[string]any{}
					}
					tc.Metadata["resultBlock"] = rb
				}
				delete(serverToolIdxByID, block.ToolUseID)
			}
		}
	}
	result.Text = strings.Join(textParts, "")
	result.Reasoning = strings.Join(reasoningParts, "")

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
	if resp.InputTransformations != nil {
		if providerMeta == nil {
			providerMeta = map[string]any{}
		}
		providerMeta["inputTransformations"] = resp.InputTransformations
	}
	if providerMeta != nil {
		result.Response.ProviderMetadata = providerMeta
		result.ProviderMetadata = map[string]map[string]any{
			"anthropic": providerMeta,
		}
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

	// Apply body transformer (e.g. Vertex removes "model", adds "anthropic_version").
	if m.opts.bodyTransformer != nil {
		body = m.opts.bodyTransformer(body)
	}

	// Determine streaming from the body (set by buildRequest).
	streaming, _ := body["stream"].(bool)

	// Build request URL.
	var reqURL string
	if m.opts.urlBuilder != nil {
		reqURL = m.opts.urlBuilder(m.opts.baseURL, m.id, streaming)
	} else {
		reqURL = m.opts.baseURL + "/v1/messages"
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	req.Header.Set("anthropic-version", apiVersion)
	req.Header.Set("anthropic-beta", betaFeatures)
	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	for k, v := range reqHeaders {
		req.Header.Set(k, v)
	}
	// Caller headers can replace the baseline, but must not remove betas
	// required by features explicitly present in this request.
	allBetas := req.Header.Get("anthropic-beta")
	if len(toolBetas) > 0 {
		seen := make(map[string]bool)
		for b := range strings.SplitSeq(allBetas, ",") {
			seen[strings.TrimSpace(b)] = true
		}
		for _, b := range toolBetas {
			if !seen[b] {
				if allBetas != "" {
					allBetas += ","
				}
				allBetas += b
				seen[b] = true
			}
		}
	}
	req.Header.Set("anthropic-beta", allBetas)

	// Set auth header last so per-request headers can never override it.
	switch m.opts.authMode {
	case AuthBearer:
		req.Header.Set("Authorization", "Bearer "+token)
	default:
		req.Header.Set("x-api-key", token)
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Bound the error-response body read so a hostile/large error payload
		// cannot exhaust memory.
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		_ = resp.Body.Close()
		errProvider := cmp.Or(m.opts.errorProvider, "anthropic")
		return nil, goai.ParseHTTPErrorWithHeaders(errProvider, resp.StatusCode, respBody, resp.Header)
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
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}
