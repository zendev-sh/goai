// Package openai provides an OpenAI language model implementation for GoAI.
//
// It supports both the Chat Completions API and the Responses API. All models
// default to Responses API (matching Vercel v2.0.89+). Chat Completions is
// available via ProviderOptions["useResponsesAPI"] = false.
//
// Usage:
//
//	model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

const defaultBaseURL = "https://api.openai.com/v1"

// Option configures the OpenAI provider.
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

// WithBaseURL overrides the default OpenAI API base URL.
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
// This enables custom transports for proxies, logging, URL rewriting,
// auth token injection, and other middleware patterns.
// Equivalent to Vercel AI SDK's `fetch` option.
// Default: http.DefaultClient.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

// Chat creates an OpenAI language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	if o.tokenSource == nil {
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPENAI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return &chatModel{
		id:   modelID,
		opts: o,
	}
}

// chatModel implements provider.LanguageModel for OpenAI.
type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature: !isReasoningModel(m.id),
		Reasoning:   isReasoningModel(m.id),
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
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: openai: WithPromptCaching is not supported and will be ignored\n")
	}
	if m.shouldUseResponsesAPI(params) {
		return m.doGenerateResponses(ctx, params)
	}
	return m.doGenerateChatCompletions(ctx, params)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: openai: WithPromptCaching is not supported and will be ignored\n")
	}
	if m.shouldUseResponsesAPI(params) {
		return m.doStreamResponses(ctx, params)
	}
	return m.doStreamChatCompletions(ctx, params)
}

// --- Chat Completions API ---

func (m *chatModel) doStreamChatCompletions(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	body := openaicompat.BuildRequest(params, m.id, true, openaicompat.RequestConfig{
		IncludeStreamOptions: true,
	})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	scanner := sse.NewScanner(resp.Body)
	go func() {
		var closeOnce sync.Once
		closeBody := func() { closeOnce.Do(func() { _ = resp.Body.Close() }) }
		defer closeBody()
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				closeBody()
			case <-done:
			}
		}()
		openaicompat.ParseStream(ctx, scanner, out)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

func (m *chatModel) doGenerateChatCompletions(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	body := openaicompat.BuildRequest(params, m.id, false, openaicompat.RequestConfig{})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return openaicompat.ParseResponse(respBody)
}

// --- Responses API ---

func (m *chatModel) doStreamResponses(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	body := buildResponsesRequest(params, m.id, true)

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/responses", body)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	go func() {
		var closeOnce sync.Once
		closeBody := func() { closeOnce.Do(func() { _ = resp.Body.Close() }) }
		// Close body on context cancellation to unblock scanner.Scan().
		// Without this, the goroutine leaks if the server stalls mid-stream.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				closeBody()
			case <-done:
			}
		}()
		// Wrap body so streamResponses' defer close calls closeBody, not raw Close,
		// preventing double-close when context cancellation races with normal completion.
		streamResponses(ctx, onceCloser{resp.Body, closeBody}, out)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

func (m *chatModel) doGenerateResponses(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	body := buildResponsesRequest(params, m.id, false)

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/responses", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return parseResponsesResult(respBody)
}

// onceCloser wraps an io.ReadCloser so that Close is idempotent via a provided
// close function. Used to prevent double-close when context cancellation races
// with the normal end-of-stream close inside streamResponses.
type onceCloser struct {
	io.Reader
	closeFn func()
}

func (o onceCloser) Close() error {
	o.closeFn()
	return nil
}

// --- HTTP helpers ---

func (m *chatModel) doHTTP(ctx context.Context, url string, body map[string]any) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Extract per-request headers before marshaling (they must not appear in the JSON body).
	reqHeaders, _ := body["_headers"].(map[string]string)
	delete(body, "_headers")

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", url, jsonBody)
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
		return nil, goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
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

// --- Model routing ---

// shouldUseResponsesAPI returns true if the model should use the Responses API.
// Item 3: Default ALL models to Responses API (matching Vercel v2.0.89+),
// unless the caller explicitly opts out via ProviderOptions["useResponsesAPI"] = false.
func (m *chatModel) shouldUseResponsesAPI(params provider.GenerateParams) bool {
	// Allow explicit override via provider options.
	if v, ok := params.ProviderOptions["useResponsesAPI"]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	// Default: all models use Responses API (matches Vercel).
	return true
}

// isReasoningModel returns true if the model is a reasoning model (o-series, gpt-5+, codex-).
// Used for capability detection (temperature, reasoning support).
func isReasoningModel(modelID string) bool {
	id := strings.ToLower(modelID)

	// o-series reasoning models (o1, o3, o4, etc.)
	if len(id) >= 2 && id[0] == 'o' && id[1] >= '0' && id[1] <= '9' {
		return true
	}

	// GPT-5+ models (except gpt-5-chat which is NOT a reasoning model per Vercel)
	if strings.HasPrefix(id, "gpt-5") && !strings.HasPrefix(id, "gpt-5-chat") {
		return true
	}

	// codex- prefix models
	if strings.HasPrefix(id, "codex-") {
		return true
	}

	return false
}
