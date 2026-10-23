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
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel          = (*chatModel)(nil)
	_ provider.CapableModel           = (*chatModel)(nil)
	_ provider.FileUploadCapableModel = (*chatModel)(nil)
)

const defaultBaseURL = "https://api.openai.com/v1"

// Option configures the OpenAI provider.
type Option func(*options)

type options struct {
	tokenSource                provider.TokenSource
	baseURL                    string
	headers                    map[string]string
	httpClient                 *http.Client
	responsesStreamIdleTimeout time.Duration
	responsesStreamAllowDone   bool
	outputFormat               string
	useMaxCompletionTokens     *bool
	errorProvider              string // provider name for error parsing (default "openai")
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

// WithUseMaxCompletionTokens forces Chat Completions requests to send
// max_completion_tokens instead of max_tokens. Reasoning models (gpt-5,
// o-series, codex) reject max_tokens outright and require
// max_completion_tokens. The OpenAI layer normally infers this from the model
// id, but callers whose wire id is not the model id (e.g. Azure deployments)
// must opt in explicitly. Nil keeps the model-id heuristic.
func WithUseMaxCompletionTokens(use bool) Option {
	return func(o *options) {
		o.useMaxCompletionTokens = &use
	}
}

// WithErrorProvider overrides the provider name used in error parsing (default
// "openai"). OpenAI-compatible hosts such as Azure should set it so a 404 for a
// missing deployment is not retried like OpenAI's own transient 404.
func WithErrorProvider(name string) Option {
	return func(o *options) {
		o.errorProvider = name
	}
}

// Chat creates an OpenAI language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{
		baseURL:                    defaultBaseURL,
		responsesStreamIdleTimeout: DefaultResponsesStreamIdleTimeout,
	}
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
		Temperature: !openaicompat.IsReasoningModel(m.id),
		Reasoning:   openaicompat.IsReasoningModel(m.id),
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
		IncludeStreamOptions:   true,
		UseMaxCompletionTokens: m.opts.useMaxCompletionTokens,
	})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}

	return openaicompat.NewSSEStream(ctx, resp.Body), nil
}

func (m *chatModel) doGenerateChatCompletions(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	body := openaicompat.BuildRequest(params, m.id, false, openaicompat.RequestConfig{
		UseMaxCompletionTokens: m.opts.useMaxCompletionTokens,
	})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := openaicompat.ReadResponseBody(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return openaicompat.ParseResponse(respBody)
}

// --- Responses API ---

func (m *chatModel) doStreamResponses(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if m.opts.responsesStreamIdleTimeout < 0 {
		return nil, fmt.Errorf("openai: Responses stream idle timeout must be non-negative: %s", m.opts.responsesStreamIdleTimeout)
	}

	body := buildResponsesRequest(params, m.id, true)

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/responses", body)
	if err != nil {
		return nil, err
	}
	// Keep resp.Request and its serialized request body out of the stream closure.
	responseBody := resp.Body
	retryHeaders := httpc.RetryHeaders(resp.Header)

	out := make(chan provider.StreamChunk, 64)
	go func() {
		streamResponsesWithConfig(ctx, responseBody, out, responsesStreamConfig{
			idleTimeout:  m.opts.responsesStreamIdleTimeout,
			allowDone:    m.opts.responsesStreamAllowDone,
			retryHeaders: retryHeaders,
		})
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

	respBody, err := openaicompat.ReadResponseBody(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return parseResponsesResult(respBody)
}

// --- HTTP helpers ---

func (m *chatModel) doHTTP(ctx context.Context, url string, body map[string]any) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	return httpc.DoJSONRequest(ctx, httpc.RequestConfig{
		URL:        url,
		Token:      token,
		Body:       body,
		Headers:    m.opts.headers,
		HTTPClient: m.opts.httpClient,
		ProviderID: cmp.Or(m.opts.errorProvider, "openai"),
	}, goai.ParseHTTPErrorWithHeaders)
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
