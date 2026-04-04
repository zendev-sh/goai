// Package bedrock provides an AWS Bedrock language model implementation for GoAI.
//
// Uses AWS SigV4 signing for authentication (no AWS SDK dependency).
//
// Usage:
//
//	model := bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0",
//		bedrock.WithAccessKey("AKIA..."),
//		bedrock.WithSecretKey("..."),
//	)
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package bedrock

import (
	"cmp"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

// ReasoningType specifies how reasoning/thinking is configured.
type ReasoningType string

const (
	// ReasoningEnabled enables extended thinking with a token budget.
	ReasoningEnabled ReasoningType = "enabled"
	// ReasoningDisabled disables reasoning.
	ReasoningDisabled ReasoningType = "disabled"
	// ReasoningAdaptive lets the model decide when to use reasoning.
	ReasoningAdaptive ReasoningType = "adaptive"
)

// ReasoningEffort controls the level of reasoning effort.
type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortMax    ReasoningEffort = "max"
)

// ReasoningConfig configures extended thinking for Bedrock models.
// Matches Vercel AI SDK's bedrockProviderOptions.reasoningConfig.
type ReasoningConfig struct {
	// Type controls whether reasoning is enabled, disabled, or adaptive.
	Type ReasoningType
	// BudgetTokens sets the token budget for reasoning (used when Type is "enabled").
	BudgetTokens int
	// MaxReasoningEffort controls the level of reasoning effort.
	MaxReasoningEffort ReasoningEffort
}

// Option configures the Bedrock provider.
type Option func(*options)

type options struct {
	region                       string
	accessKey                    string
	secretKey                    string
	sessionToken                 string
	bearerToken                  string
	baseURL                      string
	headers                      map[string]string
	httpClient                   *http.Client
	additionalModelRequestFields map[string]any
	reasoningConfig              *ReasoningConfig
	anthropicBeta                []string
}

// WithRegion sets the AWS region.
func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
	}
}

// WithAccessKey sets the AWS access key ID.
func WithAccessKey(key string) Option {
	return func(o *options) {
		o.accessKey = key
	}
}

// WithSecretKey sets the AWS secret access key.
func WithSecretKey(key string) Option {
	return func(o *options) {
		o.secretKey = key
	}
}

// WithSessionToken sets the AWS session token (for temporary credentials).
func WithSessionToken(token string) Option {
	return func(o *options) {
		o.sessionToken = token
	}
}

// WithBearerToken sets a Bearer token for Bedrock authentication, bypassing
// SigV4 signing. Matches Vercel AI SDK's bedrockOptions.bedrockOptions.token
// and the AWS_BEARER_TOKEN_BEDROCK environment variable.
func WithBearerToken(token string) Option {
	return func(o *options) {
		o.bearerToken = token
	}
}

// WithBaseURL overrides the default Bedrock endpoint (useful for testing).
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

// WithAdditionalModelRequestFields sets extra fields to merge into the request body.
// These are passed through to the Bedrock API as additionalModelRequestFields.
// Matches Vercel AI SDK's bedrockProviderOptions.additionalModelRequestFields.
func WithAdditionalModelRequestFields(fields map[string]any) Option {
	return func(o *options) {
		o.additionalModelRequestFields = fields
	}
}

// WithReasoningConfig configures extended thinking for Bedrock models.
// Matches Vercel AI SDK's bedrockProviderOptions.reasoningConfig.
//
// For Anthropic models, this maps to the "thinking" field in the request body.
// For non-Anthropic models, this maps to the "reasoningConfig" field.
func WithReasoningConfig(cfg ReasoningConfig) Option {
	return func(o *options) {
		o.reasoningConfig = &cfg
	}
}

// WithAnthropicBeta sets the Anthropic beta features to enable via the
// x-amz-bedrock-anthropic-beta header. Matches Vercel AI SDK's
// bedrockProviderOptions.anthropicBeta.
func WithAnthropicBeta(betas []string) Option {
	return func(o *options) {
		o.anthropicBeta = betas
	}
}

// Chat creates a Bedrock language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve region: explicit > AWS_REGION > AWS_DEFAULT_REGION > us-east-1.
	o.region = cmp.Or(o.region, os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"), "us-east-1")
	// Cross-region inference profiles (e.g., "us.anthropic.claude-*") must
	// be called from a matching geo's endpoint. Override region if it conflicts.
	if r := inferRegionFromModel(modelID); r != "" && !regionMatchesGeo(o.region, modelID) {
		o.region = r
	}
	// Resolve credentials from env if not set.
	o.bearerToken = cmp.Or(o.bearerToken, os.Getenv("AWS_BEARER_TOKEN_BEDROCK"))
	o.accessKey = cmp.Or(o.accessKey, os.Getenv("AWS_ACCESS_KEY_ID"))
	o.secretKey = cmp.Or(o.secretKey, os.Getenv("AWS_SECRET_ACCESS_KEY"))
	o.sessionToken = cmp.Or(o.sessionToken, os.Getenv("AWS_SESSION_TOKEN"))
	o.baseURL = cmp.Or(o.baseURL, os.Getenv("AWS_BEDROCK_BASE_URL"))
	return &chatModel{id: modelID, originalID: modelID, opts: o}
}

type chatModel struct {
	mu                sync.RWMutex // protects id and opts.region (mutated by fallback)
	id                string
	originalID        string // preserved for bare-ID-from-US fallback
	opts              options
	fallbackAttempted sync.Once
	fallbackDone      bool
}

func (m *chatModel) ModelID() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.id
}

// readIDRegion returns the current (id, region) under read lock.
func (m *chatModel) readIDRegion() (string, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.id, m.opts.region
}

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	id := m.ModelID()
	return provider.ModelCapabilities{
		Temperature: true,
		ToolCall:    modelSupportsTools(id),
		Reasoning:   bedrockSupportsThinking(id),
		Attachment:  true,
		InputModalities: provider.ModalitySet{
			Text:  true,
			Image: true,
			PDF:   true,
		},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

// bedrockSupportsThinking returns true for Bedrock model IDs that support extended thinking.
// Bedrock model IDs include the Anthropic model name (e.g., "anthropic.claude-sonnet-4-20250514-v1:0").
func bedrockSupportsThinking(modelID string) bool {
	return strings.Contains(modelID, "claude-3-7-sonnet") ||
		strings.Contains(modelID, "claude-sonnet-4") ||
		strings.Contains(modelID, "claude-opus-4")
}

// modelSupportsTools returns whether a Bedrock model supports tool_use.
// DeepSeek R1 and Titan Text Lite/Express do not support tool calling.
func modelSupportsTools(modelID string) bool {
	lower := strings.ToLower(modelID)
	// DeepSeek R1 (reasoning model) does not support tool_use on Bedrock.
	// DeepSeek V3 DOES support tool_use.
	if strings.Contains(lower, "deepseek") && strings.Contains(lower, "r1") {
		return false
	}
	// Amazon Titan Text Lite and Express do not support tool_use.
	if strings.Contains(lower, "titan-text-lite") || strings.Contains(lower, "titan-text-express") {
		return false
	}
	return true
}

const responseFormatToolName = "__json_response"

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	rfMode := params.ResponseFormat != nil
	if rfMode {
		params = injectResponseFormatTool(params)
	}

	resp, err := m.doRequest(ctx, params, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	result, parseErr := parseConverseResponse(respBody)
	if parseErr != nil {
		return nil, parseErr
	}

	if rfMode {
		extractResponseFormatResult(result)
	}

	// Populate response metadata matching Vercel AI SDK pattern:
	// id = x-amzn-requestid header, model = the model ID used.
	result.Response.ID = resp.Header.Get("X-Amzn-Requestid")
	result.Response.Model = m.ModelID()

	return result, nil
}

// injectResponseFormatTool uses the Anthropic "tool trick": inject a synthetic tool
// with the JSON schema and force the model to call it, producing structured output.
func injectResponseFormatTool(params provider.GenerateParams) provider.GenerateParams {
	p := params
	p.Tools = append([]provider.ToolDefinition{{
		Name:        responseFormatToolName,
		Description: "Return structured JSON response",
		InputSchema: params.ResponseFormat.Schema,
	}}, p.Tools...)
	p.ToolChoice = responseFormatToolName
	p.ResponseFormat = nil
	return p
}

// extractResponseFormatResult converts the synthetic tool call back to text.
func extractResponseFormatResult(result *provider.GenerateResult) {
	for i, tc := range result.ToolCalls {
		if tc.Name == responseFormatToolName {
			result.Text = string(tc.Input)
			result.ToolCalls = append(result.ToolCalls[:i], result.ToolCalls[i+1:]...)
			if len(result.ToolCalls) == 0 {
				result.FinishReason = provider.FinishStop
			}
			return
		}
	}
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	rfMode := params.ResponseFormat != nil
	if rfMode {
		params = injectResponseFormatTool(params)
	}
	resp, err := m.doRequest(ctx, params, true)
	if err != nil {
		return nil, err
	}

	responseMeta := provider.ResponseMetadata{
		ID:    resp.Header.Get("X-Amzn-Requestid"),
		Model: m.ModelID(),
	}

	innerCh := make(chan provider.StreamChunk, 64)
	go parseEventStream(ctx, resp.Body, innerCh, responseMeta, rfMode)

	// Read initial chunks to detect tool-streaming errors that arrive as
	// EventStream exceptions (e.g., nvidia.nemotron returns HTTP 200 but
	// immediately sends an exception frame about tool/streaming incompatibility).
	if len(params.Tools) > 0 {
		var buffered []provider.StreamChunk
		for chunk := range innerCh {
			buffered = append(buffered, chunk)
			if chunk.Type == provider.ChunkError && isStreamChunkToolError(chunk.Error) {
				// Drain the old stream to prevent goroutine leak.
				// Channel is closed by parseEventStream when done.
				go func() {
					for range innerCh {
					}
				}()

				// Retry without tools.
				params.Tools = nil
				params.ToolChoice = ""
				resp2, err2 := m.doRequest(ctx, params, true)
				if err2 != nil {
					return nil, err2
				}
				retryMeta := provider.ResponseMetadata{
					ID:    resp2.Header.Get("X-Amzn-Requestid"),
					Model: m.ModelID(),
				}
				out := make(chan provider.StreamChunk, 64)
				go parseEventStream(ctx, resp2.Body, out, retryMeta, rfMode)
				return &provider.StreamResult{Stream: out}, nil
			}
			// Stop peeking once we get meaningful content or a non-tool error.
			if chunk.Type == provider.ChunkText && chunk.Text != "" {
				break
			}
			if chunk.Type == provider.ChunkError {
				break // non-tool error, don't retry
			}
			if chunk.Type == provider.ChunkToolCallStreamStart || chunk.Type == provider.ChunkFinish {
				break
			}
		}

		// Pipe buffered chunks + remaining through output channel.
		// Select on ctx.Done to prevent goroutine leak if consumer abandons.
		out := make(chan provider.StreamChunk, 64)
		go func() {
			defer close(out)
			for _, chunk := range buffered {
				select {
				case out <- chunk:
				case <-ctx.Done():
					return
				}
			}
			for chunk := range innerCh {
				select {
				case out <- chunk:
				case <-ctx.Done():
					return
				}
			}
		}()
		return &provider.StreamResult{Stream: out}, nil
	}

	return &provider.StreamResult{Stream: innerCh}, nil
}

// isStreamChunkToolError checks if a stream error is about tools not being
// supported in streaming mode.
func isStreamChunkToolError(err error) bool {
	if err == nil {
		return false
	}
	return matchesToolStreamingMessage(strings.ToLower(err.Error()))
}

// doRequest builds and sends the Converse request, with automatic fallback for
// cross-region inference profiles and maxTokens limit errors.
func (m *chatModel) doRequest(ctx context.Context, params provider.GenerateParams, streaming bool) (*http.Response, error) {
	resp, err := m.buildAndSend(ctx, params, streaming)

	// Fallback chain: each entry tries a different recovery strategy.
	fallbacks := []struct {
		name    string
		applies func() bool
		mutate  func()
	}{
		{"cross-region", func() bool { return m.tryCrossRegionFallback(err) }, nil},
		{"bare-us", func() bool { return m.tryBareUSFallback(err) }, nil},
		{"maxTokens", func() bool { return isMaxTokensError(err) && params.MaxOutputTokens > 0 }, func() {
			params.MaxOutputTokens = 0
		}},
		{"tool-streaming", func() bool { return streaming && isToolStreamingError(err) && len(params.Tools) > 0 }, func() {
			params.Tools = nil
			params.ToolChoice = ""
		}},
		{"prompt-caching", func() bool { return isPromptCachingError(err) && params.PromptCaching }, func() {
			params.PromptCaching = false
		}},
	}

	for _, fb := range fallbacks {
		if err == nil || !fb.applies() {
			continue
		}
		if fb.mutate != nil {
			fb.mutate()
		}
		resp, err = m.buildAndSend(ctx, params, streaming)
	}

	return resp, err
}

// buildAndSend constructs the Converse request body and sends it.
func (m *chatModel) buildAndSend(ctx context.Context, params provider.GenerateParams, streaming bool) (*http.Response, error) {
	id, _ := m.readIDRegion()
	body := buildConverseRequest(params, id)
	m.applyBedrockOptions(body, params.Tools, params.ProviderOptions)
	if len(params.Tools) == 0 {
		delete(body, "toolConfig")
		// Bedrock Converse API requires toolConfig whenever messages contain
		// toolUse or toolResult content blocks. This happens after Esc interrupt:
		// conversation history retains tool blocks from the aborted turn, but
		// the new request may not include tools (e.g., compaction agent).
		// Scan messages and synthesize a minimal toolConfig from referenced tools.
		ensureToolConfigForHistory(body, params.Messages)
	}
	return m.doHTTP(ctx, body, streaming)
}

// isMaxTokensError checks if the error is about exceeding the model's token limit.
func isMaxTokensError(err error) bool {
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	return strings.Contains(apiErr.Message, "maximum tokens") &&
		strings.Contains(apiErr.Message, "exceeds")
}

// isPromptCachingError checks if the error indicates the model doesn't support prompt caching.
func isPromptCachingError(err error) bool {
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	msg := strings.ToLower(apiErr.Message)
	return strings.Contains(msg, "prompt caching") ||
		(strings.Contains(msg, "unsupported model") && strings.Contains(msg, "cach"))
}

// parseReasoningConfig extracts a ReasoningConfig from dynamic ProviderOptions.
// This unifies the typed (WithReasoningConfig) and dynamic (ProviderOptions map) paths
// into a single ReasoningConfig struct.
func parseReasoningConfig(opts map[string]any) *ReasoningConfig {
	rc, ok := opts["reasoningConfig"].(map[string]any)
	if !ok {
		return nil
	}
	cfg := &ReasoningConfig{}
	if t, ok := rc["type"].(string); ok {
		cfg.Type = ReasoningType(t)
	}
	switch bt := rc["budgetTokens"].(type) {
	case int:
		cfg.BudgetTokens = bt
	case float64:
		cfg.BudgetTokens = int(bt)
	}
	if e, ok := rc["maxReasoningEffort"].(string); ok {
		cfg.MaxReasoningEffort = ReasoningEffort(e)
	}
	return cfg
}

// isToolStreamingError checks if the error is about tool calling not being
// supported in streaming mode (e.g., writer.palmyra, nvidia.nemotron).
func isToolStreamingError(err error) bool {
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	return matchesToolStreamingMessage(strings.ToLower(apiErr.Message))
}

// matchesToolStreamingMessage checks if a lowercased error message indicates
// tool calling is not supported in streaming mode.
func matchesToolStreamingMessage(msg string) bool {
	if strings.Contains(msg, "mantle streaming error") {
		return true
	}
	if !strings.Contains(msg, "tool") {
		return false
	}
	hasStream := strings.Contains(msg, "streaming") || strings.Contains(msg, "stream")
	hasUnsupported := strings.Contains(msg, "not supported") || strings.Contains(msg, "doesn't support") || strings.Contains(msg, "does not support")
	return hasStream && hasUnsupported
}

// tryBareUSFallback reverts the us. prefix and tries the original bare model ID
// from us-east-1. This handles models (e.g., ai21) that don't have cross-region
// inference profiles but are available in us-east-1 with their bare ID.
func (m *chatModel) tryBareUSFallback(err error) bool {
	// Only applies if we already did a us. prefix fallback.
	m.mu.RLock()
	currentID := m.id
	originalID := m.originalID
	fallbackDone := m.fallbackDone
	m.mu.RUnlock()

	if !fallbackDone || originalID == "" || currentID == originalID {
		return false
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	if !strings.Contains(apiErr.Message, "model identifier is invalid") {
		return false
	}
	// Revert to bare ID, keep us-east-1 region.
	m.mu.Lock()
	m.id = m.originalID
	m.mu.Unlock()
	return true
}

// tryCrossRegionFallback attempts a cross-region inference profile when a bare
// model ID fails with "invalid model identifier" or "on-demand throughput isn't
// supported". Many models are only available in US regions; prepending "us."
// routes the request through a US endpoint. Returns true (and mutates m) if
// the fallback should be attempted. Only runs once per model instance.
func (m *chatModel) tryCrossRegionFallback(err error) bool {
	currentID := m.ModelID()
	if err == nil || inferRegionFromModel(currentID) != "" {
		return false // already has geo prefix, or no error
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	isInvalidModel := strings.Contains(apiErr.Message, "model identifier is invalid") ||
		strings.Contains(apiErr.Message, "on-demand throughput isn\u2019t supported") ||
		strings.Contains(apiErr.Message, "on-demand throughput isn't supported")
	if !isInvalidModel {
		return false
	}
	// Only attempt fallback once.
	m.fallbackAttempted.Do(func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.id = "us." + m.id
		m.opts.region = "us-east-1"
		m.fallbackDone = true
	})
	m.mu.RLock()
	done := m.fallbackDone
	m.mu.RUnlock()
	return done
}

// applyBedrockOptions merges Bedrock-specific options into the request body.
// For the Converse API, reasoning and additional fields go under "additionalModelRequestFields".
func (m *chatModel) applyBedrockOptions(body map[string]any, tools []provider.ToolDefinition, providerOpts map[string]any) {
	isAnthropicModel := strings.Contains(m.ModelID(), "anthropic")

	// Merge additionalModelRequestFields into the Converse API wrapper field.
	additional := maps.Clone(m.opts.additionalModelRequestFields)
	if additional == nil {
		additional = make(map[string]any)
	}

	// Resolve reasoning config: per-request ProviderOptions take precedence
	// over construction-time WithReasoningConfig(). This allows consumers to
	// pass "reasoningConfig" in ProviderOptions for per-request thinking control.
	reasoningCfg := m.opts.reasoningConfig
	if reasoningCfg == nil && providerOpts != nil {
		reasoningCfg = parseReasoningConfig(providerOpts)
	}

	// Apply reasoningConfig into additionalModelRequestFields.
	isThinkingEnabled := false
	if reasoningCfg != nil {
		cfg := reasoningCfg
		if isAnthropicModel {
			switch cfg.Type {
			case ReasoningEnabled:
				thinking := map[string]any{"type": "enabled"}
				if cfg.BudgetTokens > 0 {
					thinking["budget_tokens"] = cfg.BudgetTokens
				}
				additional["thinking"] = thinking
				isThinkingEnabled = true
			case ReasoningAdaptive:
				additional["thinking"] = map[string]any{"type": "adaptive"}
				isThinkingEnabled = true
			}

			if cfg.MaxReasoningEffort != "" {
				additional["output_config"] = map[string]any{
					"effort": string(cfg.MaxReasoningEffort),
				}
			}
		} else {
			if cfg.MaxReasoningEffort != "" {
				rc := map[string]any{
					"maxReasoningEffort": string(cfg.MaxReasoningEffort),
				}
				if cfg.Type != "" {
					rc["type"] = string(cfg.Type)
				}
				additional["reasoningConfig"] = rc
			}
		}
	}

	// Anthropic beta features go in the request body as anthropic_beta,
	// matching Vercel AI SDK behavior (not as an HTTP header).
	// Merge user-supplied betas with auto-detected tool betas.
	var allBetas []string
	allBetas = append(allBetas, m.opts.anthropicBeta...)
	for _, t := range tools {
		if t.ProviderDefinedType != "" {
			if beta := toolBetaForType(t.ProviderDefinedType); beta != "" {
				allBetas = append(allBetas, beta)
			}
		}
	}
	if len(allBetas) > 0 {
		// Deduplicate.
		seen := make(map[string]bool)
		unique := make([]string, 0, len(allBetas))
		for _, b := range allBetas {
			if !seen[b] {
				seen[b] = true
				unique = append(unique, b)
			}
		}
		additional["anthropic_beta"] = unique
	}

	if len(additional) > 0 {
		body["additionalModelRequestFields"] = additional
	}

	// When thinking is enabled, adjust inferenceConfig per Vercel parity:
	// - Remove topK, temperature, topP (Anthropic rejects them with thinking)
	// - Add budgetTokens to maxTokens (Bedrock requires maxTokens >= budget_tokens)
	if isThinkingEnabled {
		ic, ok := body["inferenceConfig"].(map[string]any)
		if !ok {
			ic = make(map[string]any)
			body["inferenceConfig"] = ic
		}
		{
			delete(ic, "topK")
			delete(ic, "temperature")
			delete(ic, "topP")

			// maxTokens must include budget_tokens for thinking-enabled models.
			if reasoningCfg != nil && reasoningCfg.BudgetTokens > 0 {
				budget := reasoningCfg.BudgetTokens
				if mt, ok := ic["maxTokens"].(int); ok {
					ic["maxTokens"] = mt + budget
				} else {
					ic["maxTokens"] = budget + 4096
				}
			}
		}
	}
}

func (m *chatModel) resolveAuth() error {
	if m.opts.bearerToken == "" && (m.opts.accessKey == "" || m.opts.secretKey == "") {
		return errors.New("bedrock: AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY required")
	}
	return nil
}

func (m *chatModel) doHTTP(ctx context.Context, body map[string]any, streaming bool) (*http.Response, error) {
	if err := m.resolveAuth(); err != nil {
		return nil, err
	}

	// Extract per-request headers from params if present.
	reqHeaders, _ := body["_headers"].(map[string]string)
	delete(body, "_headers")

	jsonBody := httpc.MustMarshalJSON(body)

	endpoint := "/converse"
	if streaming {
		endpoint = "/converse-stream"
	}

	id, region := m.readIDRegion()

	var reqURL string
	escapedID := url.PathEscape(id)
	if m.opts.baseURL != "" {
		reqURL = m.opts.baseURL + "/model/" + escapedID + endpoint
	} else {
		if !validRegion(region) {
			return nil, fmt.Errorf("bedrock: invalid AWS region %q", region)
		}
		reqURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s%s",
			region, escapedID, endpoint)
	}

	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	for k, v := range reqHeaders {
		req.Header.Set(k, v)
	}

	// Set User-Agent header.
	req.Header.Set("User-Agent", "goai/amazon-bedrock")

	// Auth: Bearer token (simple) or SigV4 (AWS standard).
	// Matches Vercel AI SDK which supports both modes.
	if m.opts.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+m.opts.bearerToken)
	} else {
		signAWSSigV4(req, jsonBody, m.opts.accessKey, m.opts.secretKey, m.opts.sessionToken, region, "bedrock")
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, goai.ParseHTTPErrorWithHeaders("bedrock", resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

func (m *chatModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}

// signAWSSigV4 signs an HTTP request using AWS Signature Version 4.
func signAWSSigV4(req *http.Request, body []byte, accessKey, secretKey, sessionToken, region, service string) {
	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("x-amz-date", amzDate)
	if sessionToken != "" {
		req.Header.Set("x-amz-security-token", sessionToken)
	}

	payloadHash := sha256Hex(body)
	req.Header.Set("x-amz-content-sha256", payloadHash)

	// Build signed headers list dynamically: always include content-type, host,
	// and all x-amz-* headers that are set on the request.
	signedHeaders := []string{"content-type", "host"}
	for h := range req.Header {
		lower := strings.ToLower(h)
		if strings.HasPrefix(lower, "x-amz-") {
			signedHeaders = append(signedHeaders, lower)
		}
	}
	slices.Sort(signedHeaders)

	var canonicalHeaders strings.Builder
	for _, h := range signedHeaders {
		val := req.Header.Get(h)
		if h == "host" {
			val = req.URL.Host
		}
		canonicalHeaders.WriteString(h + ":" + strings.TrimSpace(val) + "\n")
	}

	canonicalRequest := strings.Join([]string{
		"POST",
		uriEncodePath(req.URL.Path),
		req.URL.RawQuery,
		canonicalHeaders.String(),
		strings.Join(signedHeaders, ";"),
		payloadHash,
	}, "\n")

	credentialScope := dateStamp + "/" + region + "/" + service + "/aws4_request"
	stringToSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + credentialScope + "\n" + sha256Hex([]byte(canonicalRequest))

	signingKey := hmacSHA256(hmacSHA256(hmacSHA256(hmacSHA256(
		[]byte("AWS4"+secretKey), []byte(dateStamp)),
		[]byte(region)),
		[]byte(service)),
		[]byte("aws4_request"))

	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	auth := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		accessKey, credentialScope, strings.Join(signedHeaders, ";"), signature)
	req.Header.Set("Authorization", auth)
}

// uriEncodePath URI-encodes each path segment per AWS SigV4 rules (RFC 3986).
// Only unreserved characters (A-Z, a-z, 0-9, -, _, ., ~) remain unencoded.
// Slashes between segments are preserved.
func uriEncodePath(path string) string {
	var buf strings.Builder
	for i := range len(path) {
		c := path[i]
		if c == '/' || isUnreserved(c) {
			buf.WriteByte(c)
		} else {
			fmt.Fprintf(&buf, "%%%02X", c)
		}
	}
	return buf.String()
}

// geoRegions maps cross-region inference profile prefixes to default regions.
// "us." → must call from us-* endpoint, "eu." → eu-*, "ap." → ap-*, "global." → any.
var geoRegions = []struct {
	prefix string
	region string
}{
	{"us.", "us-east-1"},
	{"eu.", "eu-west-1"},
	{"ap.", "ap-southeast-1"},
	{"global.", "us-east-1"},
}

// inferRegionFromModel detects the required AWS region from a cross-region
// inference profile prefix. Returns empty string for non-prefixed model IDs.
func inferRegionFromModel(modelID string) string {
	for _, p := range geoRegions {
		if strings.HasPrefix(modelID, p.prefix) {
			return p.region
		}
	}
	return ""
}

// regionMatchesGeo checks whether a region belongs to the same geo as
// the model's cross-region inference profile prefix.
func regionMatchesGeo(region, modelID string) bool {
	for _, p := range geoRegions {
		if strings.HasPrefix(modelID, p.prefix) {
			geo := strings.TrimSuffix(p.prefix, ".")
			if geo == "global" {
				return true // global profiles work from any region
			}
			return strings.HasPrefix(region, geo+"-")
		}
	}
	return true // no geo prefix → any region is fine
}

// toolBetaForType maps provider-defined tool types to Anthropic beta header values.
// Keep in sync with provider/anthropic/tools.go betaForTool.
func toolBetaForType(toolType string) string {
	switch toolType {
	case "computer_20241022", "bash_20241022", "text_editor_20241022":
		return "computer-use-2024-10-22"
	case "computer_20250124", "bash_20250124", "text_editor_20250124", "text_editor_20250429":
		return "computer-use-2025-01-24"
	case "computer_20251124":
		return "computer-use-2025-11-24"
	case "text_editor_20250728":
		return "" // no beta needed (GA)
	case "code_execution_20250825":
		return "code-execution-2025-08-25"
	case "code_execution_20260120":
		return "" // no beta needed (GA)
	case "web_search_20260209", "web_fetch_20260209":
		// Bedrock does not support this beta flag via additionalModelRequestFields.
		// These tools work on Bedrock without the beta (unlike Anthropic direct API).
		return ""
	default:
		return ""
	}
}

// validRegion checks that the AWS region is safe for hostname interpolation.
var validRegionRE = regexp.MustCompile(`^[a-z][a-z0-9-]{0,62}$`)

func validRegion(region string) bool {
	return validRegionRE.MatchString(region)
}

func isUnreserved(c byte) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
		(c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~'
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}
