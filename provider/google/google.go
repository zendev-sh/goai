// Package google provides a Google Gemini language model implementation for GoAI.
//
// It uses the Gemini REST API with SSE streaming and a unique wire format.
//
// Usage:
//
//	model := google.Chat("gemini-2.5-flash", google.WithAPIKey("..."))
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package google

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/gemini"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

const defaultBaseURL = "https://generativelanguage.googleapis.com"

// Option configures the Google provider.
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

// WithBaseURL overrides the default Gemini API base URL.
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

// Chat creates a Google Gemini language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	// Support both GOOGLE_GENERATIVE_AI_API_KEY (Vercel AI SDK convention)
	// and GEMINI_API_KEY (Google's own convention / models.dev).
	if o.tokenSource == nil {
		if key := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY")); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("GOOGLE_GENERATIVE_AI_BASE_URL"); base != "" {
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

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: true,
		Reasoning:   true,
		ToolCall:    true,
		Attachment:  true,
		InputModalities: provider.ModalitySet{
			Text:  true,
			Image: true,
			Video: true,
			PDF:   true,
		},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	body, err := m.buildRequest(params)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent", m.opts.baseURL, m.id)

	resp, err := m.doHTTP(ctx, url, body, params.Headers)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return parseResponse(respBody)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	body, err := m.buildRequest(params)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", m.opts.baseURL, m.id)

	resp, err := m.doHTTP(ctx, url, body, params.Headers)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				_ = resp.Body.Close()
			case <-done:
			}
		}()
		parseSSE(ctx, resp.Body, out)
		close(done)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

// --- Request building ---

// geminiRequestBody controls JSON field order for Gemini implicit caching.
// Fields are serialized in struct order: systemInstruction and tools (stable prefix)
// come before contents (which changes per turn), maximizing cache hits.
type geminiRequestBody struct {
	SystemInstruction any `json:"systemInstruction,omitempty"`
	Tools             any `json:"tools,omitempty"`
	ToolConfig        any `json:"toolConfig,omitempty"`
	SafetySettings    any `json:"safetySettings,omitempty"`
	GenerationConfig  any `json:"generationConfig,omitempty"`
	CachedContent     any `json:"cachedContent,omitempty"`
	Labels            any `json:"labels,omitempty"`
	Contents          any `json:"contents"`
}

// googleOpts extracts the "google" sub-map from ProviderOptions.
func googleOpts(params provider.GenerateParams) map[string]any {
	if g, ok := params.ProviderOptions["google"].(map[string]any); ok {
		return g
	}
	return nil
}

func (m *chatModel) buildRequest(params provider.GenerateParams) (geminiRequestBody, error) {
	body := geminiRequestBody{}
	gopts := googleOpts(params)

	// System instruction (first for cache prefix).
	if params.System != "" {
		body.SystemInstruction = map[string]any{
			"parts": []map[string]any{
				{"text": params.System},
			},
		}
	}

	// Tools -- split into function declarations and provider-defined tools.
	// Gemini API sends these as separate entries in the tools array:
	//   [{"functionDeclarations": [...]}, {"googleSearch": {...}}, {"urlContext": {}}, ...]
	var functionDecls []map[string]any
	var providerTools []map[string]any

	for _, t := range params.Tools {
		if t.ProviderDefinedType != "" {
			// Provider-defined tool (google_search, url_context, code_execution).
			// Map ProviderDefinedType to Gemini API key format.
			providerTools = append(providerTools, googleProviderTool(t))
		} else {
			// Regular function tool.
			decl := map[string]any{
				"name":        t.Name,
				"description": t.Description,
			}
			if len(t.InputSchema) > 0 {
				var schema any
				if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
					return geminiRequestBody{}, fmt.Errorf("google: invalid tool schema for %s: %w", t.Name, err)
				}
				if schemaMap, ok := schema.(map[string]any); ok {
					schema = sanitizeGeminiSchema(schemaMap)
				}
				decl["parameters"] = schema
			}
			functionDecls = append(functionDecls, decl)
		}
	}

	var tools []map[string]any
	if len(functionDecls) > 0 {
		tools = append(tools, map[string]any{"functionDeclarations": functionDecls})
	}
	tools = append(tools, providerTools...)

	// Legacy ProviderOptions google_search -- kept for backward compat.
	if gopts != nil {
		if gs, ok := gopts["google_search"].(map[string]any); ok {
			tools = append(tools, map[string]any{"googleSearch": gs})
		}
	}

	if len(tools) > 0 {
		body.Tools = tools
	}

	// Tool choice → toolConfig.functionCallingConfig.
	toolConfig := map[string]any{}
	if params.ToolChoice != "" {
		fcc := map[string]any{}
		switch params.ToolChoice {
		case "auto":
			fcc["mode"] = "AUTO"
		case "none":
			fcc["mode"] = "NONE"
		case "required":
			fcc["mode"] = "ANY"
		default:
			fcc["mode"] = "ANY"
			fcc["allowedFunctionNames"] = []string{params.ToolChoice}
		}
		toolConfig["functionCallingConfig"] = fcc
	}

	// Legacy retrievalConfig from ProviderOptions.
	if gopts != nil {
		if rc, ok := gopts["retrievalConfig"].(map[string]any); ok {
			toolConfig["retrievalConfig"] = rc
		}
	}

	if len(toolConfig) > 0 {
		body.ToolConfig = toolConfig
	}

	// Safety settings from ProviderOptions.
	if gopts != nil {
		if ss, ok := gopts["safetySettings"]; ok {
			body.SafetySettings = ss
		}
	}

	// Generation config.
	genConfig := map[string]any{}
	if params.MaxOutputTokens > 0 {
		genConfig["maxOutputTokens"] = params.MaxOutputTokens
	}
	if params.Temperature != nil {
		genConfig["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		genConfig["topP"] = *params.TopP
	}
	if params.TopK != nil {
		genConfig["topK"] = *params.TopK
	}

	// Thinking config -- read from ProviderOptions, fall back to defaults.
	// Only models that support thinking get a default thinkingConfig.
	// Gemma models and older Gemini (1.5, 2.0) do NOT support thinking.
	thinkingDisabled := false
	if gopts != nil {
		if tc, ok := gopts["thinkingConfig"].(map[string]any); ok {
			genConfig["thinkingConfig"] = tc
		} else if disabled, ok := gopts["thinkingConfig"].(bool); ok && !disabled {
			thinkingDisabled = true
		}
	}
	if _, hasTC := genConfig["thinkingConfig"]; !hasTC && !thinkingDisabled && modelSupportsThinking(m.id) {
		// Default thinking config for thinking-capable models.
		tc := map[string]any{"includeThoughts": true}
		if strings.Contains(m.id, "gemini-3") {
			tc["thinkingLevel"] = "high"
		}
		genConfig["thinkingConfig"] = tc
	}

	if len(params.StopSequences) > 0 {
		genConfig["stopSequences"] = params.StopSequences
	}

	// Response modalities from ProviderOptions (TEXT/IMAGE).
	if gopts != nil {
		if rm, ok := gopts["responseModalities"]; ok {
			genConfig["responseModalities"] = rm
		}
	}

	// Media resolution from ProviderOptions.
	if gopts != nil {
		if mr, ok := gopts["mediaResolution"].(string); ok && mr != "" {
			genConfig["mediaResolution"] = mr
		}
	}

	// Audio timestamp from ProviderOptions.
	if gopts != nil {
		if at, ok := gopts["audioTimestamp"].(bool); ok && at {
			genConfig["audioTimestamp"] = true
		}
	}

	// Image config from ProviderOptions.
	if gopts != nil {
		if ic, ok := gopts["imageConfig"].(map[string]any); ok {
			genConfig["imageConfig"] = ic
		}
	}

	// Response format (structured output / JSON mode).
	if params.ResponseFormat != nil {
		genConfig["responseMimeType"] = "application/json"
		if len(params.ResponseFormat.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(params.ResponseFormat.Schema, &schema); err != nil {
				return geminiRequestBody{}, fmt.Errorf("google: invalid response schema: %w", err)
			}
			if schemaMap, ok := schema.(map[string]any); ok {
				schema = sanitizeGeminiSchema(schemaMap)
			}
			genConfig["responseSchema"] = schema
		}
	}

	body.GenerationConfig = genConfig

	// Cached content from ProviderOptions.
	if gopts != nil {
		if cc, ok := gopts["cachedContent"].(string); ok && cc != "" {
			body.CachedContent = cc
		}
	}

	// Labels from ProviderOptions.
	if gopts != nil {
		if lb, ok := gopts["labels"].(map[string]any); ok {
			body.Labels = lb
		}
	}

	// Messages (last, changes per turn).
	body.Contents = convertMessages(params.Messages)

	return body, nil
}

// --- Message conversion ---

func convertMessages(msgs []provider.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))

	for _, msg := range msgs {
		if msg.Role == provider.RoleSystem {
			continue // handled by systemInstruction
		}

		role := string(msg.Role)
		if role == "assistant" {
			role = "model"
		}
		if role == "tool" {
			role = "user" // Gemini uses user role for function responses
		}

		parts := make([]map[string]any, 0, len(msg.Content))
		for _, part := range msg.Content {
			switch part.Type {
			case provider.PartText:
				textPart := map[string]any{"text": part.Text}
				// Preserve thoughtSignature for multi-turn reasoning.
				if google, ok := part.ProviderOptions["google"].(map[string]any); ok {
					if sig, ok := google["thoughtSignature"].(string); ok && sig != "" {
						textPart["thoughtSignature"] = sig
					}
				}
				parts = append(parts, textPart)

			case provider.PartImage, provider.PartFile:
				mediaType, data, ok := httpc.ParseDataURL(part.URL)
				if ok {
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": mediaType,
							"data":     data,
						},
					})
				}

			case provider.PartReasoning:
				reasoningPart := map[string]any{
					"thought": true,
					"text":    part.Text,
				}
				if google, ok := part.ProviderOptions["google"].(map[string]any); ok {
					if sig, ok := google["thoughtSignature"].(string); ok && sig != "" {
						reasoningPart["thoughtSignature"] = sig
					}
				}
				parts = append(parts, reasoningPart)

			case provider.PartToolCall:
				var args any
				if len(part.ToolInput) > 0 {
					_ = json.Unmarshal(part.ToolInput, &args) // nil on error → fallback below
				}
				if args == nil {
					args = map[string]any{}
				}
				fcPart := map[string]any{
					"functionCall": map[string]any{
						"name": part.ToolName,
						"args": args,
					},
				}
				if google, ok := part.ProviderOptions["google"].(map[string]any); ok {
					if sig, ok := google["thoughtSignature"].(string); ok && sig != "" {
						fcPart["thoughtSignature"] = sig
					}
				}
				parts = append(parts, fcPart)

			case provider.PartToolResult:
				// Gemini requires functionResponse.response to be an object.
				var response any
				if err := json.Unmarshal([]byte(part.ToolOutput), &response); err != nil {
					response = map[string]any{"result": part.ToolOutput}
				}
				// Wrap non-object types.
				if _, ok := response.(map[string]any); !ok {
					response = map[string]any{"result": response}
				}
				parts = append(parts, map[string]any{
					"functionResponse": map[string]any{
						"name":     part.ToolName,
						"response": response,
					},
				})
			}
		}

		if len(parts) > 0 {
			result = append(result, map[string]any{
				"role":  role,
				"parts": parts,
			})
		}
	}

	return result
}


// --- SSE parsing ---

// groundingChunk represents a single grounding source from Gemini's groundingMetadata.
type groundingChunk struct {
	Web *struct {
		URI   string `json:"uri"`
		Title string `json:"title"`
	} `json:"web,omitempty"`
	RetrievedContext *struct {
		URI   string `json:"uri"`
		Title string `json:"title"`
	} `json:"retrievedContext,omitempty"`
}

// groundingMetadata contains grounding/citation data from Gemini responses.
type groundingMetadata struct {
	GroundingChunks []groundingChunk `json:"groundingChunks,omitempty"`
}

// geminiResponse matches the Gemini SSE response structure.
type geminiResponse struct {
	ModelVersion string `json:"modelVersion,omitempty"`
	Candidates   []struct {
		Content struct {
			Parts []struct {
				Text             string `json:"text,omitempty"`
				Thought          bool   `json:"thought,omitempty"`
				ThoughtSignature string `json:"thoughtSignature,omitempty"`
				FunctionCall     *struct {
					Name string          `json:"name"`
					Args json.RawMessage `json:"args"`
				} `json:"functionCall,omitempty"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason     string             `json:"finishReason,omitempty"`
		GroundingMetadata *groundingMetadata `json:"groundingMetadata,omitempty"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
		CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	} `json:"usageMetadata,omitempty"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func parseSSE(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
	defer close(out)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	var callIndex int

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var resp geminiResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		// Capture model version from the first chunk that has it.
		if responseMeta.Model == "" && resp.ModelVersion != "" {
			responseMeta.Model = resp.ModelVersion
		}

		// Handle error response in SSE.
		// Terminal sends: function returns immediately after, so unchecked TrySend is intentional.
		if resp.Error != nil {
			msg := resp.Error.Message
			if goai.IsOverflow(msg) {
				provider.TrySend(ctx, out, provider.StreamChunk{
					Type:  provider.ChunkError,
					Error: &goai.ContextOverflowError{Message: msg},
				})
			} else {
				provider.TrySend(ctx, out, provider.StreamChunk{
					Type:  provider.ChunkError,
					Error: &goai.APIError{Message: msg, StatusCode: resp.Error.Code},
				})
			}
			return
		}

		// Update usage.
		if resp.UsageMetadata != nil {
			usage.CacheReadTokens = resp.UsageMetadata.CachedContentTokenCount
			usage.InputTokens = resp.UsageMetadata.PromptTokenCount - usage.CacheReadTokens
			if usage.InputTokens < 0 {
				usage.InputTokens = 0
			}
			usage.ReasoningTokens = resp.UsageMetadata.ThoughtsTokenCount
			usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount - resp.UsageMetadata.ThoughtsTokenCount
			if usage.OutputTokens < 0 {
				usage.OutputTokens = 0
			}
		}

		if len(resp.Candidates) == 0 {
			continue
		}

		candidate := resp.Candidates[0]

		for _, part := range candidate.Content.Parts {
			// Build provider metadata for thoughtSignature.
			var meta map[string]any
			if part.ThoughtSignature != "" {
				meta = map[string]any{
					"google": map[string]any{
						"thoughtSignature": part.ThoughtSignature,
					},
				}
			}

			if part.FunctionCall != nil {
				// Gemini sends complete function calls (not streaming).
				argsStr := string(part.FunctionCall.Args)
				callID := fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, callIndex)
				callIndex++

				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCallStreamStart,
					ToolCallID: callID,
					ToolName:   part.FunctionCall.Name,
					Metadata:   meta,
				}) {
					return
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCall,
					ToolCallID: callID,
					ToolName:   part.FunctionCall.Name,
					ToolInput:  argsStr,
					Metadata:   meta,
				}) {
					return
				}
			} else if part.Thought {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: part.Text, Metadata: meta}) {
					return
				}
			} else if part.Text != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: part.Text, Metadata: meta}) {
					return
				}
			}
		}

		// Emit grounding sources from groundingMetadata.
		if gSources := extractGroundingSources(candidate.GroundingMetadata); len(gSources) > 0 {
			for _, src := range gSources {
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkText,
					Metadata: map[string]any{
						"source": src,
					},
				}) {
					return
				}
			}
		}

		if candidate.FinishReason != "" {
			u := usage // copy
			fr := mapFinishReason(candidate.FinishReason)
			// Gemini returns STOP even when tool calls are present.
			if fr == provider.FinishStop && callIndex > 0 {
				fr = provider.FinishToolCalls
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: fr,
				Usage:        u,
			}) {
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) // terminal send
		return
	}

	// Send final finish chunk only on clean stream completion.
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	provider.TrySend(ctx, out, provider.StreamChunk{ // terminal send
		Type:     provider.ChunkFinish,
		Usage:    usage,
		Response: responseMeta,
	})
}

// extractGroundingSources converts groundingMetadata into provider.Source entries.
func extractGroundingSources(gm *groundingMetadata) []provider.Source {
	if gm == nil || len(gm.GroundingChunks) == 0 {
		return nil
	}
	sources := make([]provider.Source, 0, len(gm.GroundingChunks))
	for i, chunk := range gm.GroundingChunks {
		if chunk.Web != nil {
			sources = append(sources, provider.Source{
				ID:    fmt.Sprintf("grounding_%d", i),
				Type:  "url",
				URL:   chunk.Web.URI,
				Title: chunk.Web.Title,
			})
		} else if chunk.RetrievedContext != nil {
			sources = append(sources, provider.Source{
				ID:    fmt.Sprintf("grounding_%d", i),
				Type:  "document",
				URL:   chunk.RetrievedContext.URI,
				Title: chunk.RetrievedContext.Title,
			})
		}
	}
	return sources
}

// modelSupportsThinking returns true if the model supports thinking/reasoning.
// Only Gemini 2.5+ and 3.x models support thinking. Gemma and older Gemini do not.
func modelSupportsThinking(modelID string) bool {
	return strings.HasPrefix(modelID, "gemini-2.5") ||
		strings.HasPrefix(modelID, "gemini-3")
}

// mapFinishReason converts Gemini finish reasons to GoAI FinishReason.
func mapFinishReason(reason string) provider.FinishReason {
	switch reason {
	case "STOP":
		return provider.FinishStop
	case "MAX_TOKENS":
		return provider.FinishLength
	case "SAFETY", "IMAGE_SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return provider.FinishContentFilter
	case "MALFORMED_FUNCTION_CALL":
		return provider.FinishError
	default:
		return provider.FinishOther
	}
}

// --- Non-streaming response parsing ---

func parseResponse(body []byte) (*provider.GenerateResult, error) {
	var resp geminiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing gemini response: %w", err)
	}

	if resp.Error != nil {
		msg := resp.Error.Message
		if goai.IsOverflow(msg) {
			return nil, &goai.ContextOverflowError{Message: msg, ResponseBody: string(body)}
		}
		return nil, &goai.APIError{Message: msg, StatusCode: resp.Error.Code}
	}

	result := &provider.GenerateResult{
		FinishReason: provider.FinishStop,
	}

	if resp.ModelVersion != "" {
		result.Response.Model = resp.ModelVersion
	}

	// Usage.
	if resp.UsageMetadata != nil {
		result.Usage.CacheReadTokens = resp.UsageMetadata.CachedContentTokenCount
		result.Usage.InputTokens = resp.UsageMetadata.PromptTokenCount - result.Usage.CacheReadTokens
		if result.Usage.InputTokens < 0 {
			result.Usage.InputTokens = 0
		}
		result.Usage.ReasoningTokens = resp.UsageMetadata.ThoughtsTokenCount
		result.Usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount - resp.UsageMetadata.ThoughtsTokenCount
		if result.Usage.OutputTokens < 0 {
			result.Usage.OutputTokens = 0
		}
		result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens
	}

	if len(resp.Candidates) == 0 {
		return result, nil
	}

	candidate := resp.Candidates[0]
	if candidate.FinishReason != "" {
		result.FinishReason = mapFinishReason(candidate.FinishReason)
	}

	var textParts []string
	var providerMeta map[string]any
	var callIndex int
	for _, part := range candidate.Content.Parts {
		if part.FunctionCall != nil {
			callID := fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, callIndex)
			callIndex++
			result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
				ID:    callID,
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			})
		} else if !part.Thought && part.Text != "" {
			textParts = append(textParts, part.Text)
		}
		// Preserve thoughtSignature for multi-turn reasoning.
		if part.ThoughtSignature != "" {
			if providerMeta == nil {
				providerMeta = map[string]any{}
			}
			// Store the last thoughtSignature -- callers can use it in subsequent turns.
			providerMeta["thoughtSignature"] = part.ThoughtSignature
		}
	}
	result.Text = strings.Join(textParts, "")

	// Attach provider metadata if we have thoughtSignatures.
	if providerMeta != nil {
		result.ProviderMetadata = map[string]map[string]any{
			"google": providerMeta,
		}
	}

	// Extract grounding sources from groundingMetadata.
	result.Sources = extractGroundingSources(candidate.GroundingMetadata)

	if len(result.ToolCalls) > 0 {
		result.FinishReason = provider.FinishToolCalls
	}

	return result, nil
}

// --- HTTP helpers ---

func (m *chatModel) doHTTP(ctx context.Context, url string, body any, perRequestHeaders map[string]string) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", url, jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", token)

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	for k, v := range perRequestHeaders {
		req.Header.Set(k, v)
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
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

// --- Schema sanitization ---

// sanitizeGeminiSchema delegates to the shared internal implementation.
func sanitizeGeminiSchema(schema map[string]any) map[string]any {
	return gemini.SanitizeSchema(schema)
}

// sanitizeImpl wraps gemini.SanitizeSchema for unit tests that exercise
// non-map inputs (nil, scalars). Map inputs delegate to SanitizeSchema;
// non-maps pass through unchanged.
var sanitizeImpl = func(obj any) any {
	m, ok := obj.(map[string]any)
	if !ok {
		return obj
	}
	return gemini.SanitizeSchema(m)
}
