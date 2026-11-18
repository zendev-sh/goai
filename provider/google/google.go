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
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/gemini"
	"github.com/zendev-sh/goai/internal/geminichat"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// Bounds for reading untrusted HTTP response bodies. Success responses are
// capped at 64 MiB; error bodies (used only to surface an error message) are
// capped at 1 MiB to prevent a malicious server from exhausting memory.
const (
	maxGoogleSuccessBodyBytes int64 = 64 << 20
	maxGoogleErrorBodyBytes   int64 = 1 << 20
)

// readSuccessBody reads a success response body, bounding it to
// maxGoogleSuccessBodyBytes and returning a clear error if the cap is exceeded.
func readSuccessBody(reader io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(reader, maxGoogleSuccessBodyBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxGoogleSuccessBodyBytes {
		return nil, fmt.Errorf("response body exceeds %d byte limit", maxGoogleSuccessBodyBytes)
	}
	return data, nil
}

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel          = (*chatModel)(nil)
	_ provider.CapableModel           = (*chatModel)(nil)
	_ provider.FileUploadCapableModel = (*chatModel)(nil)
	_ provider.LanguageModel          = (*vertexChatModel)(nil)
	_ provider.CapableModel           = (*vertexChatModel)(nil)
)

const defaultBaseURL = "https://generativelanguage.googleapis.com"

func init() {
	geminichat.RegisterFactory(newVertexChat)
}

// Option configures the Google provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	usesAPIKey  bool
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client

	// Vertex AI mode: when isVertex is set, requests route to the
	// {location}-aiplatform.googleapis.com endpoints and authenticate with a
	// Bearer OAuth token instead of the x-goog-api-key header. The native wire
	// format (serialization, SSE, thinking, grounding) is identical.
	isVertex      bool
	project       string
	location      string
	vertexBaseURL string
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.tokenSource = provider.StaticToken(key)
		o.usesAPIKey = true
	}
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) {
		o.tokenSource = ts
		o.usesAPIKey = false
	}
}

// WithVertex configures the provider to target Google Cloud Vertex AI endpoints.
func WithVertex(project, location string) Option {
	return func(o *options) {
		o.isVertex = true
		o.project = project
		o.location = location
	}
}

// WithVertexBaseURL overrides the base URL used in Vertex AI mode.
func WithVertexBaseURL(url string) Option {
	return func(o *options) {
		o.vertexBaseURL = url
	}
}

func newVertexChat(modelID string, cfg geminichat.Config) provider.LanguageModel {
	model := &chatModel{
		id: modelID,
		opts: options{
			tokenSource:   cfg.TokenSource,
			baseURL:       defaultBaseURL,
			headers:       cfg.Headers,
			httpClient:    cfg.HTTPClient,
			isVertex:      true,
			project:       cfg.Project,
			location:      cfg.Location,
			vertexBaseURL: cfg.BaseURL,
		},
	}
	return &vertexChatModel{model: model}
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
	o := resolveOptions(opts...)
	model := &chatModel{
		id:   modelID,
		opts: o,
	}
	if o.isVertex {
		return &vertexChatModel{model: model}
	}
	return model
}

// resolveOptions applies the functional options and fills auth/base URL from the
// environment, mirroring the resolution used by every capability constructor
// (Chat, Embedding, NewCacheClient).
func resolveOptions(opts ...Option) options {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	// Support both GOOGLE_GENERATIVE_AI_API_KEY (Vercel AI SDK convention)
	// and GEMINI_API_KEY (Google's own convention / models.dev).
	if !o.isVertex && o.tokenSource == nil {
		if key := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY")); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if !o.isVertex && o.baseURL == defaultBaseURL {
		if base := os.Getenv("GOOGLE_GENERATIVE_AI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return o
}

type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: true,
		Reasoning:   modelSupportsThinking(m.id),
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

// vertexChatModel deliberately does not implement FileUploadCapableModel.
// Vertex native Gemini endpoints do not support the Gemini Developer API's
// Files API used by chatModel.FileUploader.
type vertexChatModel struct {
	model *chatModel
}

func (m *vertexChatModel) ModelID() string { return m.model.ModelID() }

func (m *vertexChatModel) Capabilities() provider.ModelCapabilities {
	caps := m.model.Capabilities()
	caps.FileUpload = false
	return caps
}

func (m *vertexChatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	return m.model.DoGenerate(ctx, params)
}

func (m *vertexChatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	return m.model.DoStream(ctx, params)
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: google: WithPromptCaching is not supported and will be ignored\n")
	}
	body, err := m.buildRequest(params)
	if err != nil {
		return nil, err
	}
	reqURL, err := m.endpointURL("generateContent", "")
	if err != nil {
		return nil, err
	}

	resp, err := m.doHTTP(ctx, reqURL, body, params.Headers)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := readSuccessBody(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return parseResponse(respBody)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: google: WithPromptCaching is not supported and will be ignored\n")
	}
	body, err := m.buildRequest(params)
	if err != nil {
		return nil, err
	}
	reqURL, err := m.endpointURL("streamGenerateContent", "?alt=sse")
	if err != nil {
		return nil, err
	}

	resp, err := m.doHTTP(ctx, reqURL, body, params.Headers)
	if err != nil {
		return nil, err
	}
	// Keep resp.Request and its serialized request body out of the stream closure.
	responseBody := resp.Body

	return provider.RunStream(ctx, responseBody, func(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
		parseSSE(ctx, body, out)
	}), nil
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
	Contents          any `json:"contents"`
}

// googleOpts extracts the "google" sub-map from ProviderOptions.
func googleOpts(params provider.GenerateParams) map[string]any {
	if g, ok := params.ProviderOptions["google"].(map[string]any); ok {
		return g
	}
	return nil
}

// buildSystemInstruction returns the systemInstruction wire object for the
// given system prompt, or nil when empty (omitempty drops it). Shared by chat
// requests and cachedContents creation so a cached prefix matches the request.
func buildSystemInstruction(system string) any {
	if system == "" {
		return nil
	}
	return map[string]any{
		"parts": []map[string]any{
			{"text": system},
		},
	}
}

// buildToolsAndConfig serializes the tools array and toolConfig from a tool set,
// tool choice and provider options. It is the single source of truth for Gemini
// tool serialization, shared by chat requests (buildRequest) and cachedContents
// creation (cache.go) so a cached prefix is byte-identical to what a request
// would send. Returns nil payloads when there is nothing to send.
//
// Tools are split into function declarations and provider-defined tools; the
// Gemini API sends them as separate entries in the tools array:
//
//	[{"functionDeclarations": [...]}, {"googleSearch": {...}}, {"urlContext": {}}, ...]
func buildToolsAndConfig(toolDefs []provider.ToolDefinition, toolChoice string, providerOptions map[string]any) (toolsPayload any, toolConfigPayload any, err error) {
	var gopts map[string]any
	if g, ok := providerOptions["google"].(map[string]any); ok {
		gopts = g
	}

	var functionDecls []map[string]any
	var providerTools []map[string]any
	for _, t := range toolDefs {
		if t.ProviderDefinedType != "" {
			// Provider-defined tool (google_search, url_context, code_execution).
			providerTools = append(providerTools, googleProviderTool(t))
			continue
		}
		// Regular function tool.
		decl := map[string]any{
			"name":        t.Name,
			"description": t.Description,
		}
		if len(t.InputSchema) > 0 {
			var schema any
			if uerr := json.Unmarshal(t.InputSchema, &schema); uerr != nil {
				return nil, nil, fmt.Errorf("google: invalid tool schema for %s: %w", t.Name, uerr)
			}
			if schemaMap, ok := schema.(map[string]any); ok {
				schema = sanitizeGeminiSchema(schemaMap)
			}
			decl["parameters"] = schema
		}
		functionDecls = append(functionDecls, decl)
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
		toolsPayload = tools
	}

	// Tool choice → toolConfig.functionCallingConfig.
	toolConfig := map[string]any{}
	if toolChoice != "" {
		fcc := map[string]any{}
		switch toolChoice {
		case "auto":
			fcc["mode"] = "AUTO"
		case "none":
			fcc["mode"] = "NONE"
		case "required":
			fcc["mode"] = "ANY"
		default:
			fcc["mode"] = "ANY"
			fcc["allowedFunctionNames"] = []string{toolChoice}
		}
		toolConfig["functionCallingConfig"] = fcc
	}
	// Legacy retrievalConfig from ProviderOptions.
	if gopts != nil {
		if rc, ok := gopts["retrievalConfig"].(map[string]any); ok {
			toolConfig["retrievalConfig"] = rc
		}
	}
	// Gemini 3.x supports this additional toolConfig field. Accept only the
	// documented boolean from the Google namespace; SDK-derived fields above
	// must not be overridden by raw provider options.
	if gopts != nil {
		if tc, ok := gopts["toolConfig"].(map[string]any); ok {
			if include, ok := tc["includeServerSideToolInvocations"].(bool); ok {
				toolConfig["includeServerSideToolInvocations"] = include
			}
		}
	}

	if len(toolConfig) > 0 {
		toolConfigPayload = toolConfig
	}

	return toolsPayload, toolConfigPayload, nil
}

func (m *chatModel) buildRequest(params provider.GenerateParams) (geminiRequestBody, error) {
	body := geminiRequestBody{}
	gopts := googleOpts(params)

	// System instruction (first for cache prefix).
	body.SystemInstruction = buildSystemInstruction(params.System)

	// Tools and toolConfig — serialized by the shared helper so an explicit
	// cachedContents resource holds a byte-identical prefix (see cache.go).
	tools, toolConfig, err := buildToolsAndConfig(params.Tools, params.ToolChoice, params.ProviderOptions)
	if err != nil {
		return geminiRequestBody{}, err
	}
	body.Tools = tools
	body.ToolConfig = toolConfig

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

	// Image config from ProviderOptions. imageSize is not an official
	// ImageConfig field -- it belongs at the top level of generationConfig,
	// so it is hoisted out of imageConfig.
	if gopts != nil {
		if ic, ok := gopts["imageConfig"].(map[string]any); ok {
			hoisted := make(map[string]any, len(ic))
			for k, v := range ic {
				if k == "imageSize" {
					genConfig["imageSize"] = v
					continue
				}
				hoisted[k] = v
			}
			genConfig["imageConfig"] = hoisted
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

	// Messages (last, changes per turn).
	body.Contents = convertMessages(params.Messages)

	return body, nil
}

// --- Message conversion ---

func convertMessages(msgs []provider.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))
	lastWasTool := false

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

			case provider.PartImage:
				if part.RemoteRef != nil {
					// Remote reference to an uploaded file/image: emit fileData.
					// Fall back to inlineData when the raw bytes are available.
					if len(part.RemoteRef.Data) > 0 {
						parts = append(parts, map[string]any{
							"inlineData": map[string]any{
								"mimeType": part.RemoteRef.MediaType,
								"data":     base64.StdEncoding.EncodeToString(part.RemoteRef.Data),
							},
						})
					} else {
						parts = append(parts, map[string]any{
							"fileData": map[string]any{
								"fileUri":  part.RemoteRef.URI,
								"mimeType": part.RemoteRef.MediaType,
							},
						})
					}
					break
				}
				mediaType, data, ok := httpc.ParseDataURL(part.URL)
				if ok {
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": mediaType,
							"data":     data,
						},
					})
				}

			case provider.PartFile:
				if p := filePartToContent(part); p != nil {
					parts = append(parts, p)
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
				functionCall := map[string]any{
					"name": part.ToolName,
					"args": args,
				}
				if part.ToolCallID != "" {
					functionCall["id"] = part.ToolCallID
				}
				fcPart := map[string]any{
					"functionCall": functionCall,
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
				functionResponse := map[string]any{
					"name":     part.ToolName,
					"response": response,
				}
				if part.ToolCallID != "" {
					functionResponse["id"] = part.ToolCallID
				}
				parts = append(parts, map[string]any{"functionResponse": functionResponse})
			}
		}

		if len(parts) > 0 {
			content := map[string]any{
				"role":  role,
				"parts": parts,
			}
			if msg.Role == provider.RoleTool && lastWasTool && len(result) > 0 {
				existing, _ := result[len(result)-1]["parts"].([]map[string]any)
				result[len(result)-1]["parts"] = append(existing, parts...)
			} else {
				result = append(result, content)
			}
			lastWasTool = msg.Role == provider.RoleTool
		} else {
			lastWasTool = false
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
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason      string             `json:"finishReason,omitempty"`
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

// geminiPart is a single content part in a Gemini candidate. Besides text,
// thought and functionCall parts, Gemini can return richer part types
// (inlineData, executableCode, codeExecutionResult, fileData, videoMetadata)
// that must not be silently dropped.
type geminiPart struct {
	Text             string `json:"text,omitempty"`
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	FunctionCall     *struct {
		ID   string          `json:"id,omitempty"`
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	} `json:"functionCall,omitempty"`
	InlineData *struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"`
	} `json:"inlineData,omitempty"`
	ExecutableCode *struct {
		Code     string `json:"code"`
		Language string `json:"language"`
	} `json:"executableCode,omitempty"`
	CodeExecutionResult *struct {
		Output string `json:"output"`
	} `json:"codeExecutionResult,omitempty"`
	FileData *struct {
		FileURI  string `json:"fileUri"`
		MimeType string `json:"mimeType"`
	} `json:"fileData,omitempty"`
	VideoMetadata *struct {
		VideoDuration string `json:"videoDuration"`
	} `json:"videoMetadata,omitempty"`
}

// extraPart returns a structured map for the non-text, non-function parts
// Gemini can emit (inlineData, executableCode, codeExecutionResult, fileData,
// videoMetadata), or nil when the part is none of those. It lets callers
// surface these parts instead of dropping them.
func (p geminiPart) extraPart() map[string]any {
	switch {
	case p.InlineData != nil:
		return map[string]any{"inlineData": p.InlineData}
	case p.ExecutableCode != nil:
		return map[string]any{"executableCode": p.ExecutableCode}
	case p.CodeExecutionResult != nil:
		return map[string]any{"codeExecutionResult": p.CodeExecutionResult}
	case p.FileData != nil:
		return map[string]any{"fileData": p.FileData}
	case p.VideoMetadata != nil:
		return map[string]any{"videoMetadata": p.VideoMetadata}
	}
	return nil
}

func parseSSE(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
	defer close(out)

	sseScanner := sse.NewScanner(body)

	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	var callIndex int

	for data, ok := sseScanner.Next(); ok; data, ok = sseScanner.Next() {

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
					Error: &goai.ContextOverflowError{Message: msg, ResponseBody: data},
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
			// candidatesTokenCount is the visible output; thinking is reported
			// separately in thoughtsTokenCount, so subtracting it zeroed the
			// output whenever thoughts outgrew the answer.
			usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount
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
				callID := part.FunctionCall.ID
				if callID == "" {
					callID = fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, callIndex)
				}
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
			} else if extra := part.extraPart(); extra != nil {
				// Surface non-text parts (inlineData, executableCode,
				// codeExecutionResult, fileData, videoMetadata) instead of
				// dropping them. They arrive as a text chunk carrying the
				// structured part under metadata.google.part.
				extraMeta := map[string]any{"google": map[string]any{"part": extra}}
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Metadata: extraMeta}) {
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

	if err := sseScanner.Err(); err != nil {
		provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) // terminal send
		return
	}

	// Send final finish chunk only on clean stream completion.
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens + usage.ReasoningTokens
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
		result.Usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount
		result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens + result.Usage.ReasoningTokens
	}

	if len(resp.Candidates) == 0 {
		return result, nil
	}

	candidate := resp.Candidates[0]
	if candidate.FinishReason != "" {
		result.FinishReason = mapFinishReason(candidate.FinishReason)
	}

	var textParts []string
	var reasoningParts []string
	var providerMeta map[string]any
	var extraParts []map[string]any
	var callIndex int
	for _, part := range candidate.Content.Parts {
		if part.FunctionCall != nil {
			callID := part.FunctionCall.ID
			if callID == "" {
				callID = fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, callIndex)
			}
			callIndex++
			tc := provider.ToolCall{
				ID:    callID,
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			}
			if part.ThoughtSignature != "" {
				tc.Metadata = map[string]any{
					"google": map[string]any{
						"thoughtSignature": part.ThoughtSignature,
					},
				}
			}
			result.ToolCalls = append(result.ToolCalls, tc)
		} else if part.Thought && part.Text != "" {
			reasoningParts = append(reasoningParts, part.Text)
		} else if !part.Thought && part.Text != "" {
			textParts = append(textParts, part.Text)
		} else if extra := part.extraPart(); extra != nil {
			// Preserve non-text parts (inlineData, executableCode,
			// codeExecutionResult, fileData, videoMetadata) rather than
			// dropping them. They surface under metadata.google.parts.
			extraParts = append(extraParts, extra)
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
	result.Reasoning = strings.Join(reasoningParts, "")

	// Attach provider metadata if we have thoughtSignatures.
	if providerMeta != nil {
		result.ProviderMetadata = map[string]map[string]any{
			"google": providerMeta,
		}
	}
	// Attach non-text parts (inlineData, executableCode, codeExecutionResult,
	// fileData, videoMetadata) so they are not dropped from the result.
	if len(extraParts) > 0 {
		if result.ProviderMetadata == nil {
			result.ProviderMetadata = map[string]map[string]any{}
		}
		gm := result.ProviderMetadata["google"]
		if gm == nil {
			gm = map[string]any{}
			result.ProviderMetadata["google"] = gm
		}
		gm["parts"] = extraParts
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
	return doGoogleJSON(ctx, m.opts, http.MethodPost, url, body, perRequestHeaders)
}

// doGoogleJSON sends a JSON request to the Gemini REST API with the right auth
// (Bearer for Vertex, x-goog-api-key otherwise) and maps any non-2xx status to
// a goai error. Shared by chat generation and cachedContents management so they
// authenticate and route identically.
func doGoogleJSON(ctx context.Context, o options, method, url string, body any, perRequestHeaders map[string]string) (*http.Response, error) {
	token, err := resolveAuthToken(ctx, o)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, method, url, jsonBody)
	req.Header.Set("Content-Type", "application/json")
	// Apply caller-supplied headers first, then set the credential LAST so a
	// caller's header named like the auth header cannot override the real
	// credential (auth wins).
	for k, v := range o.headers {
		req.Header.Set(k, v)
	}
	for k, v := range perRequestHeaders {
		req.Header.Set(k, v)
	}
	if o.isVertex {
		req.Header.Set("Authorization", "Bearer "+token)
	} else {
		req.Header.Set("x-goog-api-key", token)
	}

	resp, err := httpc.Do(httpClientFor(o), req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxGoogleErrorBodyBytes))
		_ = resp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

func httpClientFor(o options) *http.Client {
	if o.httpClient != nil {
		return o.httpClient
	}
	return http.DefaultClient
}

// vertexHost returns the aiplatform host for a region. The "global" region has
// no location prefix on the hostname.
func vertexHost(location string) string {
	if location == "global" {
		return "aiplatform.googleapis.com"
	}
	return location + "-aiplatform.googleapis.com"
}

// endpointURL builds the request URL for an action (generateContent /
// streamGenerateContent). In Vertex mode it targets the aiplatform publisher
// endpoint; otherwise the generativelanguage base URL.
func (m *chatModel) endpointURL(action, query string) (string, error) {
	if !m.opts.isVertex {
		return fmt.Sprintf("%s/v1beta/models/%s:%s%s", m.opts.baseURL, url.PathEscape(m.id), action, query), nil
	}
	if !validGCPIdentifier(m.opts.project) {
		return "", fmt.Errorf("google: invalid vertex project %q", m.opts.project)
	}
	if !validGCPIdentifier(m.opts.location) {
		return "", fmt.Errorf("google: invalid vertex location %q", m.opts.location)
	}
	if base := strings.TrimRight(m.opts.vertexBaseURL, "/"); base != "" {
		return fmt.Sprintf("%s/%s:%s%s", base, url.PathEscape(m.id), action, query), nil
	}
	return fmt.Sprintf("https://%s/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:%s%s",
		vertexHost(m.opts.location), m.opts.project, m.opts.location, url.PathEscape(m.id), action, query), nil
}

// validGCPIdentifier guards project/location before hostname interpolation
// (anti-SSRF). Allows standard projects (my-project-123), domain-scoped
// (example.com:proj) and regions (us-east5); blocks /, \, @, .., whitespace.
var validGCPIdentifierRE = regexp.MustCompile(`^[a-z0-9][a-z0-9.:_-]{0,127}$`)

func validGCPIdentifier(s string) bool {
	return validGCPIdentifierRE.MatchString(s) && !strings.Contains(s, "..")
}

func resolveAuthToken(ctx context.Context, o options) (string, error) {
	if o.isVertex && o.usesAPIKey {
		return "", errors.New("google: WithVertex requires an OAuth token source; API keys are not supported")
	}
	if o.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return o.tokenSource.Token(ctx)
}

func validateNonChatOptions(o options) error {
	if o.isVertex {
		return errors.New("google: WithVertex is only supported by Chat; use provider/vertex for other model types")
	}
	return nil
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
