// Package cohere provides a Cohere language model and embedding implementation for GoAI.
//
// Uses Cohere's native Chat and Embed APIs.
//
// Usage:
//
//	model := cohere.Chat("command-r-plus", cohere.WithAPIKey("..."))
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
//
//	embedModel := cohere.Embedding("embed-v4.0", cohere.WithAPIKey("..."))
//	result, err := goai.Embed(ctx, embedModel, "hello world")
package cohere

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
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel  = (*chatModel)(nil)
	_ provider.CapableModel   = (*chatModel)(nil)
	_ provider.EmbeddingModel = (*embeddingModel)(nil)
)

const defaultBaseURL = "https://api.cohere.com/v2"

// Option configures the Cohere provider.
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

// WithBaseURL overrides the default Cohere API base URL.
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

// Chat creates a Cohere language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	resolveEnv(&o)
	return &chatModel{id: modelID, opts: o}
}

// Embedding creates a Cohere embedding model for the given model ID.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	resolveEnv(&o)
	return &embeddingModel{id: modelID, opts: o}
}

// resolveEnv applies environment variable fallbacks for Cohere options.
func resolveEnv(o *options) {
	if o.tokenSource == nil {
		if key := os.Getenv("COHERE_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("COHERE_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
}

// --- Chat Model ---

type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature:      true,
		ToolCall:         true,
		Reasoning:        true,
		InputModalities:  provider.ModalitySet{Text: true},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	body := buildChatRequest(params, m.id, false)

	resp, err := m.doHTTP(ctx, "/chat", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return parseChatResponse(respBody)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	body := buildChatRequest(params, m.id, true)

	resp, err := m.doHTTP(ctx, "/chat", body)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		// Close body on context cancellation to unblock parseChatStream.
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
		parseChatStream(ctx, resp.Body, out)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

func (m *chatModel) doHTTP(ctx context.Context, path string, body map[string]any) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Extract per-request headers before marshaling.
	reqHeaders, _ := body["_headers"].(map[string]string)
	delete(body, "_headers")

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+path, jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

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
		return nil, goai.ParseHTTPErrorWithHeaders("cohere", resp.StatusCode, respBody, resp.Header)
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

// --- Embedding Model ---

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

func (m *embeddingModel) MaxValuesPerCall() int { return 96 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Default input_type is "search_document"; override via ProviderOptions.
	inputType := "search_document"
	if v, ok := params.ProviderOptions["inputType"]; ok {
		if s, ok := v.(string); ok {
			inputType = s
		}
	}

	body := map[string]any{
		"model":      m.id,
		"texts":      values,
		"input_type": inputType,
	}

	// Optional truncate parameter (NONE, START, END).
	if v, ok := params.ProviderOptions["truncate"]; ok {
		if s, ok := v.(string); ok {
			body["truncate"] = s
		}
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+"/embed", jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, goai.ParseHTTPErrorWithHeaders("cohere", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Embeddings struct {
			Float [][]float64 `json:"float"`
		} `json:"embeddings"`
		Meta struct {
			BilledUnits struct {
				InputTokens int `json:"input_tokens"`
			} `json:"billed_units"`
		} `json:"meta"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("cohere: parsing response: %w", err)
	}

	return &provider.EmbedResult{
		Embeddings: result.Embeddings.Float,
		Usage: provider.Usage{
			InputTokens: result.Meta.BilledUnits.InputTokens,
			TotalTokens: result.Meta.BilledUnits.InputTokens,
		},
	}, nil
}

func (m *embeddingModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *embeddingModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}

// --- Request/Response mapping ---

func buildChatRequest(params provider.GenerateParams, modelID string, streaming bool) map[string]any {
	body := map[string]any{
		"model":  modelID,
		"stream": streaming,
	}

	// Convert messages to Cohere format.
	var msgs []map[string]any
	if params.System != "" {
		msgs = append(msgs, map[string]any{
			"role":    "system",
			"content": params.System,
		})
	}
	for _, msg := range params.Messages {
		role := string(msg.Role)
		if role == "tool" {
			for _, part := range msg.Content {
				if part.Type == provider.PartToolResult {
					msgs = append(msgs, map[string]any{
						"role":         "tool",
						"tool_call_id": part.ToolCallID,
						"content":      part.ToolOutput,
					})
				}
			}
			continue
		}

		m := map[string]any{"role": role}
		var textParts []string
		var toolCalls []map[string]any
		for _, part := range msg.Content {
			switch part.Type {
			case provider.PartText:
				textParts = append(textParts, part.Text)
			case provider.PartToolCall:
				tc := map[string]any{
					"id":   part.ToolCallID,
					"type": "function",
					"function": map[string]any{
						"name":      part.ToolName,
						"arguments": string(part.ToolInput),
					},
				}
				toolCalls = append(toolCalls, tc)
			}
		}
		if len(textParts) > 0 {
			m["content"] = strings.Join(textParts, "\n")
		}
		if len(toolCalls) > 0 {
			m["tool_calls"] = toolCalls
			if len(textParts) == 0 {
				m["content"] = ""
			}
		}
		msgs = append(msgs, m)
	}
	body["messages"] = msgs

	// Tools.
	if len(params.Tools) > 0 {
		var tools []map[string]any
		for _, tool := range params.Tools {
			var schema any
			if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
				schema = map[string]any{}
			}
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        tool.Name,
					"description": tool.Description,
					"parameters":  schema,
				},
			})
		}
		body["tools"] = tools
	}

	// Tool choice mapping (Cohere v2 uses OpenAI-compatible tool_choice).
	if params.ToolChoice != "" {
		switch params.ToolChoice {
		case "auto", "none", "required":
			body["tool_choice"] = params.ToolChoice
		default:
			// Specific tool name.
			body["tool_choice"] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": params.ToolChoice,
				},
			}
		}
	}

	if len(params.StopSequences) > 0 {
		body["stop_sequences"] = params.StopSequences
	}

	if params.MaxOutputTokens > 0 {
		body["max_tokens"] = params.MaxOutputTokens
	}
	if params.Temperature != nil {
		body["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		body["p"] = *params.TopP
	}

	// Thinking / reasoning support for command-a-reasoning models.
	// Read from ProviderOptions["thinking"] -- matches Vercel AI SDK convention.
	// Accepts: {type: "enabled"/"disabled", budgetTokens: N}
	if thinking, ok := params.ProviderOptions["thinking"]; ok {
		if tm, ok := thinking.(map[string]any); ok {
			thinkingReq := map[string]any{}
			if t, ok := tm["type"]; ok {
				thinkingReq["type"] = t
			} else {
				thinkingReq["type"] = "enabled"
			}
			if budget, ok := tm["budgetTokens"]; ok {
				thinkingReq["token_budget"] = budget
			}
			body["thinking"] = thinkingReq
		}
	}

	// Response format (structured output / JSON mode).
	if params.ResponseFormat != nil {
		if len(params.ResponseFormat.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(params.ResponseFormat.Schema, &schema); err == nil {
				body["response_format"] = map[string]any{
					"type": "json_object",
					"json_schema": map[string]any{
						"name":   params.ResponseFormat.Name,
						"schema": schema,
					},
				}
			}
		} else {
			body["response_format"] = map[string]any{"type": "json_object"}
		}
	}

	// Per-request headers (extracted in doHTTP before marshaling).
	if len(params.Headers) > 0 {
		body["_headers"] = params.Headers
	}

	return body
}

// cohereCitation represents a citation from Cohere's response.
type cohereCitation struct {
	Start   int    `json:"start"`
	End     int    `json:"end"`
	Text    string `json:"text"`
	Sources []struct {
		Document *struct {
			ID    string `json:"id"`
			Title string `json:"title"`
			Text  string `json:"text"`
		} `json:"document,omitempty"`
	} `json:"sources"`
}

func parseChatResponse(body []byte) (*provider.GenerateResult, error) {
	var resp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Message struct {
			Role    string `json:"role"`
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			} `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			Citations []cohereCitation `json:"citations"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
		Usage        struct {
			BilledUnits struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"billed_units"`
			Tokens struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("cohere: parsing response: %w", err)
	}

	result := &provider.GenerateResult{
		Response: provider.ResponseMetadata{ID: resp.ID, Model: resp.Model},
	}

	// Extract text from content blocks. Reasoning (thinking) blocks are
	// stored in ProviderMetadata so callers can access them if needed.
	var textParts []string
	var reasoningParts []string
	for _, c := range resp.Message.Content {
		switch c.Type {
		case "text":
			if c.Text != "" {
				textParts = append(textParts, c.Text)
			}
		case "thinking":
			if c.Thinking != "" {
				reasoningParts = append(reasoningParts, c.Thinking)
			}
		}
	}
	result.Text = strings.Join(textParts, "")
	if len(reasoningParts) > 0 {
		result.ProviderMetadata = map[string]map[string]any{
			"cohere": {"reasoning": strings.Join(reasoningParts, "")},
		}
	}

	// Extract tool calls.
	for _, tc := range resp.Message.ToolCalls {
		result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}

	// Map finish reason.
	result.FinishReason = cmp.Or(mapFinishReason(resp.FinishReason), provider.FinishOther)

	// Usage (prefer tokens, fallback to billed_units).
	result.Usage.InputTokens = resp.Usage.Tokens.InputTokens
	result.Usage.OutputTokens = resp.Usage.Tokens.OutputTokens
	if result.Usage.InputTokens == 0 {
		result.Usage.InputTokens = resp.Usage.BilledUnits.InputTokens
	}
	if result.Usage.OutputTokens == 0 {
		result.Usage.OutputTokens = resp.Usage.BilledUnits.OutputTokens
	}
	result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens

	// Extract citations.
	result.Sources = extractCohereCitations(resp.Message.Citations)

	return result, nil
}

// extractCohereCitations converts Cohere citation objects into provider.Source entries.
func extractCohereCitations(citations []cohereCitation) []provider.Source {
	if len(citations) == 0 {
		return nil
	}
	var sources []provider.Source
	for i, cit := range citations {
		for _, src := range cit.Sources {
			if src.Document != nil {
				s := provider.Source{
					ID:         src.Document.ID,
					Type:       "document",
					Title:      src.Document.Title,
					StartIndex: cit.Start,
					EndIndex:   cit.End,
					ProviderMetadata: map[string]any{
						"cohere": map[string]any{
							"citationIndex": i,
							"text":          cit.Text,
						},
					},
				}
				sources = append(sources, s)
			}
		}
	}
	return sources
}

// pendingToolCall tracks a tool call being accumulated across streaming events.
type pendingToolCall struct {
	id   string
	name string
	args strings.Builder
}

func parseChatStream(ctx context.Context, body io.Reader, out chan<- provider.StreamChunk) {
	defer close(out)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var pending *pendingToolCall
	var isReasoning bool
	var streamCitations []cohereCitation

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Delta struct {
				Message struct {
					Content json.RawMessage `json:"content"`
					ToolCalls struct {
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
				Usage        struct {
					Tokens struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
					} `json:"tokens"`
				} `json:"usage"`
			} `json:"delta"`
			Usage struct {
				BilledUnits struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"billed_units"`
				Tokens struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"tokens"`
			} `json:"usage"`
			FinishReason string `json:"finish_reason"`
		}

		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content-start":
			// Check if this is a thinking content block.
			var ct struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(event.Delta.Message.Content, &ct) == nil && ct.Type == "thinking" {
				isReasoning = true
			}

		case "content-delta":
			// Try thinking field first, then text.
			var thinking struct {
				Thinking string `json:"thinking"`
			}
			var text struct {
				Text string `json:"text"`
			}
			if json.Unmarshal(event.Delta.Message.Content, &thinking) == nil && thinking.Thinking != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkReasoning,
					Text: thinking.Thinking,
				}) {
					return
				}
			} else if json.Unmarshal(event.Delta.Message.Content, &text) == nil && text.Text != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkText,
					Text: text.Text,
				}) {
					return
				}
			}

		case "content-end":
			if isReasoning {
				isReasoning = false
			}

		case "tool-call-start":
			pending = &pendingToolCall{
				id:   event.Delta.Message.ToolCalls.ID,
				name: event.Delta.Message.ToolCalls.Function.Name,
			}
			// Write initial arguments if any.
			if args := event.Delta.Message.ToolCalls.Function.Arguments; args != "" {
				pending.args.WriteString(args)
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:       provider.ChunkToolCallStreamStart,
				ToolCallID: pending.id,
				ToolName:   pending.name,
			}) {
				return
			}

		case "tool-call-delta":
			if pending != nil {
				pending.args.WriteString(event.Delta.Message.ToolCalls.Function.Arguments)
			}

		case "tool-call-end":
			if pending != nil {
				args := strings.TrimSpace(pending.args.String())
				if args == "" || args == "null" {
					args = "{}"
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCall,
					ToolCallID: pending.id,
					ToolName:   pending.name,
					ToolInput:  args,
				}) {
					return
				}
				pending = nil
			}

		case "citation-start":
			// Cohere streaming citation event -- parse the citation data from delta.
			var citDelta struct {
				Message struct {
					Citations cohereCitation `json:"citations"`
				} `json:"message"`
			}
			// Re-parse the delta to get citation data.
			var raw struct {
				Delta json.RawMessage `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &raw) == nil && raw.Delta != nil {
				if json.Unmarshal(raw.Delta, &citDelta) == nil {
					streamCitations = append(streamCitations, citDelta.Message.Citations)
				}
			}

		case "message-end":
			fr := cmp.Or(mapFinishReason(event.FinishReason), mapFinishReason(event.Delta.FinishReason), provider.FinishOther)

			usage := provider.Usage{
				InputTokens:  event.Delta.Usage.Tokens.InputTokens,
				OutputTokens: event.Delta.Usage.Tokens.OutputTokens,
			}
			// Fallback to top-level usage fields.
			if usage.InputTokens == 0 {
				usage.InputTokens = event.Usage.Tokens.InputTokens
			}
			if usage.OutputTokens == 0 {
				usage.OutputTokens = event.Usage.Tokens.OutputTokens
			}
			if usage.InputTokens == 0 {
				usage.InputTokens = event.Usage.BilledUnits.InputTokens
			}
			if usage.OutputTokens == 0 {
				usage.OutputTokens = event.Usage.BilledUnits.OutputTokens
			}

			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: fr,
			}) {
				return
			}
			usage.TotalTokens = usage.InputTokens + usage.OutputTokens
			finishChunk := provider.StreamChunk{
				Type:         provider.ChunkFinish,
				FinishReason: fr,
				Usage:        usage,
				Response:     provider.ResponseMetadata{},
			}
			if sources := extractCohereCitations(streamCitations); len(sources) > 0 {
				finishChunk.Metadata = map[string]any{"sources": sources}
			}
			if !provider.TrySend(ctx, out, finishChunk) {
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		provider.TrySend(ctx, out, provider.StreamChunk{ // terminal send
			Type:  provider.ChunkError,
			Error: fmt.Errorf("reading stream: %w", err),
		})
	}
}

func mapFinishReason(reason string) provider.FinishReason {
	switch reason {
	case "COMPLETE", "END_TURN":
		return provider.FinishStop
	case "TOOL_CALL":
		return provider.FinishToolCalls
	case "MAX_TOKENS":
		return provider.FinishLength
	case "ERROR":
		return provider.FinishError
	case "":
		return ""
	default:
		return provider.FinishOther
	}
}
