// Package ollama provides a native Ollama language model implementation for GoAI.
//
// This package communicates directly with Ollama's /api/* endpoints using
// net/http and encoding/json. No third-party Ollama SDK is required.
//
// Features:
//   - Think (extended reasoning) mode via ProviderOptions["think"] bool.
//   - Streaming tool calls forwarded as ChunkToolCallStreamStart + ChunkToolCall.
//   - Native embedding via EmbeddingModel returning float64 vectors.
//
// # Basic usage
//
//	// Chat model
//	model := ollama.Chat("llama3.2")
//	result, err := model.DoGenerate(ctx, provider.GenerateParams{...})
//
//	// Chat model with think mode
//	result, err := model.DoGenerate(ctx, provider.GenerateParams{
//	    ProviderOptions: map[string]any{"think": true},
//	    ...
//	})
//
//	// Embedding model
//	em := ollama.Embedding("nomic-embed-text")
//	res, err := em.DoEmbed(ctx, []string{"hello"}, provider.EmbedParams{})
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	goai "github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel  = (*Model)(nil)
	_ provider.EmbeddingModel = (*EmbeddingModel)(nil)
)

const (
	// defaultBaseURL is the default Ollama server address.
	defaultBaseURL = "http://localhost:11434"

	// defaultMaxValuesPerCall is the maximum number of texts that may be sent
	// to Ollama's /api/embed endpoint in one call.
	defaultMaxValuesPerCall = 2048

	// streamBufSize is the channel buffer size for streaming chunks.
	streamBufSize = 32

	// maxEmbedResponseBytes caps the embedding response body read to bound
	// memory use on an unexpectedly large or malicious response.
	maxEmbedResponseBytes = 128 << 20 // 128 MiB

	// maxErrorBodyBytes caps how much of a non-200 response body is read into
	// the returned error to avoid unbounded memory use.
	maxErrorBodyBytes = 64 << 10 // 64 KiB
)

// --- Wire types ---

// ollamaChatRequest is the wire format for POST /api/chat.
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Think    bool            `json:"think"`
	Tools    []ollamaTool    `json:"tools,omitempty"`
	Options  map[string]any  `json:"options,omitempty"`
	// Format requests structured output. Ollama accepts either the string
	// "json" for generic JSON mode or a JSON Schema object for constrained
	// output, so it is carried as a raw JSON value.
	Format json.RawMessage `json:"format,omitempty"`
}

// ollamaMessage is a single message in the Ollama chat wire format.
type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Thinking   string           `json:"thinking,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// ollamaToolCall is a tool call in the Ollama wire format.
type ollamaToolCall struct {
	Function ollamaToolCallFunction `json:"function"`
}

// ollamaToolCallFunction holds the name and arguments of a tool call.
type ollamaToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ollamaTool is a tool definition in the Ollama wire format.
type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

// ollamaToolFunction holds the definition of a tool function.
type ollamaToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ollamaChatResponse is one NDJSON line in the /api/chat streaming response.
type ollamaChatResponse struct {
	Model           string        `json:"model"`
	Message         ollamaMessage `json:"message"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
	EvalCount       int           `json:"eval_count,omitempty"`
}

// ollamaEmbedRequest is the wire format for POST /api/embed.
type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaEmbedResponse is the wire format response from /api/embed.
type ollamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

// --- Options ---

// Option configures a Chat or Embedding model.
type Option func(*options)

type options struct {
	baseURL    string
	headers    map[string]string
	httpClient *http.Client
}

// WithBaseURL overrides the default Ollama server base URL (default: http://localhost:11434).
func WithBaseURL(rawURL string) Option {
	return func(o *options) {
		o.baseURL = rawURL
	}
}

// WithHeaders sets additional HTTP headers to include in every request.
// The map is copied so later mutations by the caller do not affect the model.
func WithHeaders(h map[string]string) Option {
	return func(o *options) {
		if len(h) == 0 {
			o.headers = nil
			return
		}
		cp := make(map[string]string, len(h))
		for k, v := range h {
			cp[k] = v
		}
		o.headers = cp
	}
}

// WithHTTPClient replaces the default http.Client used for requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

// buildOptions applies Option functions over the zero-value options struct.
func buildOptions(opts []Option) options {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// --- Model ---

// Model implements provider.LanguageModel using the native Ollama /api/chat endpoint.
//
// Create instances via [Chat]; do not construct directly.
type Model struct {
	baseURL    string
	modelID    string
	headers    map[string]string
	httpClient *http.Client
}

// ModelID returns the Ollama model identifier (e.g. "llama3.2", "qwen3:30b-a3b").
func (m *Model) ModelID() string {
	return m.modelID
}

// Capabilities declares the features supported by Ollama chat models.
func (m *Model) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: true,
		ToolCall:    true,
		InputModalities: provider.ModalitySet{
			Text: true,
		},
		OutputModalities: provider.ModalitySet{
			Text: true,
		},
	}
}

// Chat creates an Ollama language model for the given model ID.
func Chat(modelID string, opts ...Option) *Model {
	o := buildOptions(opts)
	return &Model{
		baseURL:    o.baseURL,
		modelID:    modelID,
		headers:    o.headers,
		httpClient: o.httpClient,
	}
}

// DoGenerate performs a non-streaming generation request and returns the
// accumulated response once the model has finished generating.
//
// Returned fields:
//   - Text: full generated text (concatenated from all response chunks).
//   - Reasoning: thinking content when ProviderOptions["think"] is true.
//   - ToolCalls: any tool calls the model requested.
//   - FinishReason: stop, tool-calls, or other.
//   - Usage: prompt and completion token counts.
//   - Response.Model: the model Ollama actually used.
func (m *Model) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	req, err := buildChatRequest(m.modelID, params, false)
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}

	resp, err := m.doPost(ctx, "/api/chat", req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var (
		lastResp    ollamaChatResponse
		textParts   []string
		reasonParts []string
		toolCalls   []provider.ToolCall
	)

	// Read NDJSON with sse.Scanner: it uses a bufio.Reader (no 64 KiB
	// per-line limit like bufio.Scanner) while bounding each line to
	// sse.MaxLineSize so a malicious response cannot exhaust memory.
	scanner := sse.NewScanner(resp.Body)
	for {
		line, ok := scanner.NextLine()
		if !ok {
			break
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		var chunk ollamaChatResponse
		if decErr := json.Unmarshal([]byte(trimmed), &chunk); decErr != nil {
			return nil, fmt.Errorf("ollama: decode response: %w", decErr)
		}
		lastResp = chunk

		if chunk.Message.Content != "" {
			textParts = append(textParts, chunk.Message.Content)
		}
		if chunk.Message.Thinking != "" {
			reasonParts = append(reasonParts, chunk.Message.Thinking)
		}
		for _, tc := range chunk.Message.ToolCalls {
			raw := tc.Function.Arguments
			if raw == nil {
				raw = json.RawMessage("{}")
			}
			toolCalls = append(toolCalls, provider.ToolCall{
				Name:  tc.Function.Name,
				Input: raw,
			})
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return nil, fmt.Errorf("ollama: reading response: %w", scanErr)
	}

	return &provider.GenerateResult{
		Text:         strings.Join(textParts, ""),
		Reasoning:    strings.Join(reasonParts, ""),
		ToolCalls:    toolCalls,
		FinishReason: mapFinishReason(lastResp.DoneReason),
		Usage: provider.Usage{
			InputTokens:  lastResp.PromptEvalCount,
			OutputTokens: lastResp.EvalCount,
		},
		Response: provider.ResponseMetadata{
			Model: lastResp.Model,
		},
	}, nil
}

// DoStream performs a streaming generation request and returns a channel of
// provider.StreamChunk events.
//
// Chunk types emitted:
//   - ChunkText: a fragment of generated text.
//   - ChunkReasoning: a fragment of reasoning text (when think mode is enabled).
//   - ChunkToolCallStreamStart: signals start of a tool call.
//   - ChunkToolCall: emitted immediately after ChunkToolCallStreamStart with full input.
//   - ChunkFinish: sent once after the stream ends, carrying usage and finish reason.
//   - ChunkError: sent if the underlying request returns an error; no ChunkFinish follows.
func (m *Model) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	req, err := buildChatRequest(m.modelID, params, true)
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}

	resp, err := m.doPost(ctx, "/api/chat", req)
	if err != nil {
		return nil, err
	}

	ch := make(chan provider.StreamChunk, streamBufSize)

	go func() {
		defer close(ch)
		defer func() { _ = resp.Body.Close() }()

		var lastResp ollamaChatResponse

		// Read NDJSON with sse.Scanner: no 64 KiB per-line limit (unlike
		// bufio.Scanner) while still bounding each line to sse.MaxLineSize.
		scanner := sse.NewScanner(resp.Body)
		for {
			line, ok := scanner.NextLine()
			if !ok {
				break
			}
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			var chunk ollamaChatResponse
			if decErr := json.Unmarshal([]byte(trimmed), &chunk); decErr != nil {
				provider.TrySend(ctx, ch, provider.StreamChunk{
					Type:  provider.ChunkError,
					Error: fmt.Errorf("ollama: decode stream chunk: %w", decErr),
				})
				return
			}
			lastResp = chunk

			if chunk.Message.Content != "" {
				if !provider.TrySend(ctx, ch, provider.StreamChunk{
					Type: provider.ChunkText,
					Text: chunk.Message.Content,
				}) {
					return
				}
			}

			if chunk.Message.Thinking != "" {
				if !provider.TrySend(ctx, ch, provider.StreamChunk{
					Type: provider.ChunkReasoning,
					Text: chunk.Message.Thinking,
				}) {
					return
				}
			}

			for _, tc := range chunk.Message.ToolCalls {
				raw := tc.Function.Arguments
				if raw == nil {
					raw = json.RawMessage("{}")
				}
				// Emit start signal first.
				if !provider.TrySend(ctx, ch, provider.StreamChunk{
					Type:     provider.ChunkToolCallStreamStart,
					ToolName: tc.Function.Name,
				}) {
					return
				}
				// Ollama delivers complete tool calls in one chunk; emit the full call immediately.
				if !provider.TrySend(ctx, ch, provider.StreamChunk{
					Type:      provider.ChunkToolCall,
					ToolName:  tc.Function.Name,
					ToolInput: string(raw),
				}) {
					return
				}
			}
		}
		if scanErr := scanner.Err(); scanErr != nil {
			provider.TrySend(ctx, ch, provider.StreamChunk{
				Type:  provider.ChunkError,
				Error: fmt.Errorf("ollama: reading stream: %w", scanErr),
			})
			return
		}

		provider.TrySend(ctx, ch, provider.StreamChunk{
			Type:         provider.ChunkFinish,
			FinishReason: mapFinishReason(lastResp.DoneReason),
			Usage: provider.Usage{
				InputTokens:  lastResp.PromptEvalCount,
				OutputTokens: lastResp.EvalCount,
			},
		})
	}()

	return &provider.StreamResult{Stream: ch}, nil
}

// --- EmbeddingModel ---

// EmbeddingModel implements provider.EmbeddingModel using the native Ollama /api/embed endpoint.
//
// Create instances via [Embedding]; do not construct directly.
type EmbeddingModel struct {
	baseURL    string
	modelID    string
	headers    map[string]string
	httpClient *http.Client
}

// ModelID returns the Ollama embedding model identifier (e.g. "nomic-embed-text").
func (e *EmbeddingModel) ModelID() string {
	return e.modelID
}

// MaxValuesPerCall returns the maximum number of texts that may be sent per call.
func (e *EmbeddingModel) MaxValuesPerCall() int {
	return defaultMaxValuesPerCall
}

// DoEmbed generates embeddings for the given text values using the native
// Ollama /api/embed endpoint.
func (e *EmbeddingModel) DoEmbed(ctx context.Context, values []string, _ provider.EmbedParams) (*provider.EmbedResult, error) {
	reqBody := ollamaEmbedRequest{
		Model: e.modelID,
		Input: values,
	}

	resp, err := sendPost(ctx, e.baseURL, "/api/embed", e.headers, e.httpClient, reqBody)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxEmbedResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("ollama: read embed response: %w", err)
	}

	var embedResp ollamaEmbedResponse
	if err = json.Unmarshal(body, &embedResp); err != nil {
		return nil, fmt.Errorf("ollama: decode embed response: %w", err)
	}

	return &provider.EmbedResult{
		Embeddings: embedResp.Embeddings,
		Usage: provider.Usage{
			InputTokens: embedResp.PromptEvalCount,
		},
		Response: provider.ResponseMetadata{
			Model: embedResp.Model,
		},
	}, nil
}

// Embedding creates an Ollama embedding model for the given model ID.
func Embedding(modelID string, opts ...Option) *EmbeddingModel {
	o := buildOptions(opts)
	return &EmbeddingModel{
		baseURL:    o.baseURL,
		modelID:    modelID,
		headers:    o.headers,
		httpClient: o.httpClient,
	}
}

// --- HTTP helpers ---

// doPost sends a JSON POST request to the given path on the model's base URL.
func (m *Model) doPost(ctx context.Context, path string, body any) (*http.Response, error) {
	return sendPost(ctx, m.baseURL, path, m.headers, m.httpClient, body)
}

// sendPost is the shared HTTP POST helper used by both Model and EmbeddingModel.
func sendPost(ctx context.Context, baseURL, path string, headers map[string]string, httpClient *http.Client, body any) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ollama: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	client := httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		_ = resp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("ollama", resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

// --- Request building ---

// buildChatRequest converts GoAI GenerateParams into an ollamaChatRequest.
func buildChatRequest(modelID string, params provider.GenerateParams, streaming bool) (*ollamaChatRequest, error) {
	msgs, err := convertMessages(params)
	if err != nil {
		return nil, err
	}

	tools, err := convertTools(params.Tools)
	if err != nil {
		return nil, err
	}

	thinkVal := extractThink(params.ProviderOptions)

	return &ollamaChatRequest{
		Model:    modelID,
		Messages: msgs,
		Tools:    tools,
		Stream:   streaming,
		Options:  buildOllamaOptions(params),
		Think:    thinkVal,
		Format:   buildFormat(params.ResponseFormat),
	}, nil
}

// buildFormat maps a GoAI ResponseFormat to Ollama's chat "format" field.
// A schema constrains output to that JSON Schema; a format with no schema
// requests generic JSON mode. Returns nil when no response format is set.
func buildFormat(rf *provider.ResponseFormat) json.RawMessage {
	if rf == nil {
		return nil
	}
	if len(rf.Schema) > 0 {
		return rf.Schema
	}
	return json.RawMessage(`"json"`)
}

// convertMessages prepends an optional system prompt and converts all GoAI
// messages to Ollama wire messages.
func convertMessages(params provider.GenerateParams) ([]ollamaMessage, error) {
	var msgs []ollamaMessage

	if params.System != "" {
		msgs = append(msgs, ollamaMessage{
			Role:    "system",
			Content: params.System,
		})
	}

	for _, msg := range params.Messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, converted...)
	}

	return msgs, nil
}

// convertMessage converts a single GoAI Message into one or more Ollama wire messages.
func convertMessage(msg provider.Message) ([]ollamaMessage, error) {
	switch msg.Role {
	case provider.RoleSystem:
		return []ollamaMessage{{
			Role:    "system",
			Content: concatText(msg.Content),
		}}, nil

	case provider.RoleUser:
		return []ollamaMessage{{
			Role:    "user",
			Content: concatText(msg.Content),
		}}, nil

	case provider.RoleAssistant:
		var text, thinking string
		var toolCalls []ollamaToolCall

		for _, part := range msg.Content {
			switch part.Type {
			case provider.PartText:
				text += part.Text
			case provider.PartReasoning:
				// Carry reasoning in the native thinking field rather than
				// merging it into content, so it round-trips correctly.
				thinking += part.Text
			case provider.PartToolCall:
				raw := part.ToolInput
				if raw == nil {
					raw = json.RawMessage("{}")
				}
				toolCalls = append(toolCalls, ollamaToolCall{
					Function: ollamaToolCallFunction{
						Name:      part.ToolName,
						Arguments: raw,
					},
				})
			}
		}

		return []ollamaMessage{{
			Role:      "assistant",
			Content:   text,
			Thinking:  thinking,
			ToolCalls: toolCalls,
		}}, nil

	case provider.RoleTool:
		var result []ollamaMessage
		for _, part := range msg.Content {
			if part.Type == provider.PartToolResult {
				result = append(result, ollamaMessage{
					Role:       "tool",
					Content:    part.ToolOutput,
					ToolCallID: part.ToolCallID,
				})
			}
		}
		return result, nil
	}

	return nil, fmt.Errorf("ollama: unsupported message role: %s", msg.Role)
}

// concatText concatenates all PartText parts in a message content slice.
func concatText(parts []provider.Part) string {
	var sb strings.Builder
	for _, part := range parts {
		if part.Type == provider.PartText {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// convertTools converts GoAI ToolDefinitions to Ollama wire tool definitions.
func convertTools(defs []provider.ToolDefinition) ([]ollamaTool, error) {
	if len(defs) == 0 {
		return nil, nil
	}

	tools := make([]ollamaTool, 0, len(defs))
	for _, def := range defs {
		var params map[string]any
		if err := json.Unmarshal(def.InputSchema, &params); err != nil {
			return nil, fmt.Errorf("ollama: unmarshal input schema for tool %s: %w", def.Name, err)
		}
		tools = append(tools, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  params,
			},
		})
	}

	return tools, nil
}

// buildOllamaOptions maps GoAI GenerateParams sampling fields to the Ollama options map.
func buildOllamaOptions(params provider.GenerateParams) map[string]any {
	opts := map[string]any{}

	if params.Temperature != nil {
		opts["temperature"] = float32(*params.Temperature)
	}
	if params.TopP != nil {
		opts["top_p"] = float32(*params.TopP)
	}
	if params.TopK != nil {
		opts["top_k"] = *params.TopK
	}
	if params.Seed != nil {
		opts["seed"] = *params.Seed
	}
	if len(params.StopSequences) > 0 {
		opts["stop"] = params.StopSequences
	}
	if params.MaxOutputTokens > 0 {
		opts["num_predict"] = params.MaxOutputTokens
	}

	if len(opts) == 0 {
		return nil
	}

	return opts
}

// extractThink reads ProviderOptions["think"] and returns the bool value.
// Defaults to false when absent or not a bool.
func extractThink(providerOptions map[string]any) bool {
	if val, ok := providerOptions["think"]; ok {
		if boolVal, isBool := val.(bool); isBool {
			return boolVal
		}
	}
	return false
}

// mapFinishReason maps Ollama done reasons to GoAI FinishReason constants.
func mapFinishReason(doneReason string) provider.FinishReason {
	switch doneReason {
	case "stop":
		return provider.FinishStop
	case "tool_calls":
		return provider.FinishToolCalls
	default:
		return provider.FinishOther
	}
}
