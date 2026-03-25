// Package vertex provides a Google Cloud Vertex AI language model implementation for GoAI.
//
// Uses the OpenAI-compatible endpoint provided by Vertex AI.
//
// Usage:
//
//	model := vertex.Chat("gemini-2.5-pro",
//		vertex.WithTokenSource(gcpTokenSource),
//	)
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package vertex

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/gemini"
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

// Option configures the Vertex AI provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	project     string
	location    string
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
// When using an API key, requests are routed to the Gemini API endpoint
// instead of Vertex AI (no project/location needed).
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.tokenSource = &apiKeyTokenSource{key: key}
	}
}

// WithTokenSource sets a dynamic token source (e.g. GCP service account).
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) {
		o.tokenSource = ts
	}
}

// WithProject sets the GCP project ID.
func WithProject(project string) Option {
	return func(o *options) {
		o.project = project
	}
}

// WithLocation sets the GCP region (default: us-central1).
func WithLocation(location string) Option {
	return func(o *options) {
		o.location = location
	}
}

// WithBaseURL overrides the default Vertex AI endpoint.
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

// resolveOpts applies options and resolves project/location from env vars.
//
// Auth resolution (in priority order):
//  1. Explicit WithTokenSource / WithAPIKey
//  2. ADC auto-detect (gcloud, service account, GCE metadata)
//  3. Fallback to API key from GOOGLE_API_KEY / GEMINI_API_KEY env vars
//     (redirects to Gemini API endpoint -- no project/location needed)
func resolveOpts(opts []Option) options {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve project: explicit > GOOGLE_VERTEX_PROJECT (Vercel) > GOOGLE_CLOUD_PROJECT > GCLOUD_PROJECT.
	o.project = cmp.Or(o.project, os.Getenv("GOOGLE_VERTEX_PROJECT"), os.Getenv("GOOGLE_CLOUD_PROJECT"), os.Getenv("GCLOUD_PROJECT"))
	// Resolve location: explicit > GOOGLE_VERTEX_LOCATION (Vercel) > GOOGLE_CLOUD_LOCATION > us-central1.
	o.location = cmp.Or(o.location, os.Getenv("GOOGLE_VERTEX_LOCATION"), os.Getenv("GOOGLE_CLOUD_LOCATION"), "us-central1")
	// Resolve base URL from env if not overridden.
	o.baseURL = cmp.Or(o.baseURL, os.Getenv("GOOGLE_VERTEX_BASE_URL"))
	// Auto-resolve auth if no explicit token source and no custom baseURL.
	// Custom baseURL (e.g. testing, proxy) may not need auth.
	if o.tokenSource == nil && o.baseURL == "" {
		o.tokenSource = autoTokenSource(o.project != "")
	}
	return o
}

// autoTokenSource resolves auth credentials automatically.
//
// When hasProject is true (Vertex mode): prefer ADC over API key, since Vertex
// endpoints require OAuth/ADC and API keys route to the Gemini API instead.
// When hasProject is false: prefer API key (simpler setup, works with Gemini API).
func autoTokenSource(hasProject bool) provider.TokenSource {
	if hasProject {
		// Vertex mode: ADC required, API keys route to the Gemini API instead.
		ts, err := ADCTokenSource(context.Background())
		if err != nil {
			return &failingTokenSource{err: err}
		}
		return ts
	}
	// No project: API key first (routes to Gemini API), ADC as fallback.
	for _, env := range []string{"GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"} {
		if key := os.Getenv(env); key != "" {
			return &apiKeyTokenSource{key: key}
		}
	}
	ts, err := ADCTokenSource(context.Background())
	if err != nil {
		// Return a token source that will report the error on first use.
		return &failingTokenSource{err: err}
	}
	return ts
}

// failingTokenSource reports a credential resolution error on first use.
type failingTokenSource struct {
	err error
}

func (s *failingTokenSource) Token(_ context.Context) (string, error) {
	return "", s.err
}

// apiKeyTokenSource is a marker type that signals URL builders to use
// the Gemini API endpoint (?key=...) instead of Vertex (Bearer token).
type apiKeyTokenSource struct {
	key string
}

func (s *apiKeyTokenSource) Token(_ context.Context) (string, error) {
	return s.key, nil
}

// nativeBaseURL builds the base URL for native (non-OpenAI-compat) endpoints.
// For Vertex: /publishers/google. For API key fallback: generativelanguage.googleapis.com.
func nativeBaseURL(o options) string {
	if o.baseURL != "" {
		return strings.TrimRight(o.baseURL, "/")
	}
	// API key → Gemini API endpoint.
	if _, ok := o.tokenSource.(*apiKeyTokenSource); ok {
		return "https://generativelanguage.googleapis.com/v1beta"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google",
		o.location, o.project, o.location)
}

// setAuth sets the Authorization header. API keys use Bearer auth (same as OAuth tokens).
// For native Gemini API endpoints (:predict), the ?key= param is set in nativeURL instead.
func setAuth(ctx context.Context, req *http.Request, ts provider.TokenSource) error {
	if ts == nil {
		return nil
	}
	token, err := ts.Token(ctx)
	if err != nil {
		return fmt.Errorf("resolving auth token: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	return nil
}

// nativeURL builds a full URL for native endpoints (embedding/image :predict).
// For API key auth, appends ?key= (native Gemini API requires query param, not Bearer).
// For OAuth/ADC, uses Bearer header (set by setAuth).
func nativeURL(o options, modelPath string) (string, error) {
	// Validate location/project before hostname interpolation to prevent SSRF.
	// Skip empty values; they are caught by the project=="" check below.
	if o.baseURL == "" {
		if _, ok := o.tokenSource.(*apiKeyTokenSource); !ok {
			if o.location != "" && !validGCPIdentifier(o.location) {
				return "", fmt.Errorf("vertex: invalid location %q", o.location)
			}
			if o.project != "" && !validGCPIdentifier(o.project) {
				return "", fmt.Errorf("vertex: invalid project %q", o.project)
			}
		}
	}
	base := nativeBaseURL(o)
	reqURL := fmt.Sprintf("%s/%s", base, modelPath)
	if aks, ok := o.tokenSource.(*apiKeyTokenSource); ok {
		reqURL += "?key=" + url.QueryEscape(aks.key)
	} else if o.project == "" && o.baseURL == "" {
		return "", fmt.Errorf("vertex: GOOGLE_CLOUD_PROJECT required (or set GOOGLE_API_KEY for Gemini API fallback)")
	}
	return reqURL, nil
}

// isNativeAPIKeyAuth returns true if using API key with native endpoints.
// Native endpoints use ?key= param (already in URL), not Bearer header.
func isNativeAPIKeyAuth(ts provider.TokenSource) bool {
	_, ok := ts.(*apiKeyTokenSource)
	return ok
}

// Chat creates a Vertex AI language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := resolveOpts(opts)
	return &chatModel{id: modelID, opts: o}
}

type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature:      true,
		ToolCall:         true,
		InputModalities:  provider.ModalitySet{Text: true, Image: true},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

// wireModelID returns the model ID for the wire format.
// Vertex endpoint requires "google/" prefix; Gemini API fallback uses bare model ID.
func (m *chatModel) wireModelID() string {
	if _, ok := m.opts.tokenSource.(*apiKeyTokenSource); ok {
		return m.id // Gemini API: bare model name
	}
	if !strings.Contains(m.id, "/") {
		return "google/" + m.id // Vertex: add publisher prefix
	}
	return m.id
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	sanitizeToolSchemas(&params)
	stripGeminiProviderOptions(&params)
	body := openaicompat.BuildRequest(params, m.wireModelID(), false, openaicompat.RequestConfig{})

	resp, err := m.doHTTP(ctx, body)
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

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	sanitizeToolSchemas(&params)
	stripGeminiProviderOptions(&params)
	body := openaicompat.BuildRequest(params, m.wireModelID(), true, openaicompat.RequestConfig{
		IncludeStreamOptions: true,
	})

	resp, err := m.doHTTP(ctx, body)
	if err != nil {
		return nil, err
	}

	out := make(chan provider.StreamChunk, 64)
	scanner := sse.NewScanner(resp.Body)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				_ = resp.Body.Close()
			case <-done:
			}
		}()
		defer close(done)
		openaicompat.ParseStream(ctx, scanner, out)
	}()

	return &provider.StreamResult{Stream: out}, nil
}

func (m *chatModel) doHTTP(ctx context.Context, body map[string]any) (*http.Response, error) {
	reqHeaders, _ := body["_headers"].(map[string]string)
	delete(body, "_headers")

	jsonBody := httpc.MustMarshalJSON(body)

	url, err := m.resolveURL(ctx, "/chat/completions")
	if err != nil {
		return nil, err
	}

	req := httpc.MustNewRequest(ctx, "POST", url, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	if m.opts.tokenSource != nil {
		if err := setAuth(ctx, req, m.opts.tokenSource); err != nil {
			return nil, err
		}
	}

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
		return nil, goai.ParseHTTPErrorWithHeaders("vertex", resp.StatusCode, respBody, resp.Header)
	}

	return resp, nil
}

// resolveURL builds the full request URL. If using an API key (no project/ADC),
// redirects to the Gemini API endpoint. Otherwise uses the Vertex AI endpoint.
func (m *chatModel) resolveURL(ctx context.Context, path string) (string, error) {
	if m.opts.baseURL != "" {
		return strings.TrimRight(m.opts.baseURL, "/") + path, nil
	}

	// API key → Gemini API OpenAI-compat endpoint (uses Bearer auth, not ?key=).
	if _, ok := m.opts.tokenSource.(*apiKeyTokenSource); ok {
		return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/openai%s", path), nil
	}

	// Vertex AI endpoint (requires project).
	if m.opts.project == "" {
		return "", fmt.Errorf("vertex: GOOGLE_CLOUD_PROJECT required (or set GOOGLE_API_KEY for Gemini API fallback)")
	}
	// Validate location/project before hostname interpolation to prevent SSRF.
	if !validGCPIdentifier(m.opts.location) {
		return "", fmt.Errorf("vertex: invalid location %q", m.opts.location)
	}
	if !validGCPIdentifier(m.opts.project) {
		return "", fmt.Errorf("vertex: invalid project %q", m.opts.project)
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/endpoints/openapi%s",
		m.opts.location, m.opts.project, m.opts.location, path), nil
}

func (m *chatModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}

// stripGeminiProviderOptions removes Gemini-native provider options that are
// not understood by the OpenAI-compatible endpoint. These options (thinkingConfig,
// etc.) are set by consumers for the native Google API but would cause 400 errors
// when passed through to the OpenAI-compat wire format.
func stripGeminiProviderOptions(params *provider.GenerateParams) {
	if params.ProviderOptions == nil {
		return
	}
	// Copy to avoid mutating the caller's map.
	params.ProviderOptions = maps.Clone(params.ProviderOptions)
	// thinkingConfig is a Gemini-native concept -- the OpenAI-compat endpoint
	// uses reasoning_effort instead (if supported).
	delete(params.ProviderOptions, "thinkingConfig")
}

// jsonMarshalFunc is swappable for testing error paths.
var jsonMarshalFunc = json.Marshal

// sanitizeToolSchemas applies the full Gemini schema sanitization to all tool
// input schemas. Vertex AI's OpenAI-compatible endpoint has the same restrictions
// as the native Gemini API: no additionalProperties, no properties/required on
// non-object types, enum values must be strings, array items must have type, etc.
// Delegates to the google provider's SanitizeGeminiSchema for consistency.
func sanitizeToolSchemas(params *provider.GenerateParams) {
	// Copy to avoid mutating the caller's tools slice.
	params.Tools = slices.Clone(params.Tools)
	for i, t := range params.Tools {
		if len(t.InputSchema) == 0 {
			continue
		}
		var schema map[string]any
		if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
			continue // best-effort: send original schema if it can't be parsed
		}
		schema = gemini.SanitizeSchema(schema)
		cleaned, err := jsonMarshalFunc(schema)
		if err != nil {
			continue // best-effort: send original schema if sanitized version can't be marshaled
		}
		params.Tools[i].InputSchema = cleaned
	}
}

// validGCPIdentifier checks that a GCP location or project ID is safe for URL interpolation.
// Allows standard projects (my-project-123), domain-scoped projects (example.com:my-project),
// and locations (us-central1). Blocks SSRF characters (/, \, @, .., whitespace).
var validGCPIdentifierRE = regexp.MustCompile(`^[a-z0-9][a-z0-9.:_-]{0,127}$`)

func validGCPIdentifier(s string) bool {
	if !validGCPIdentifierRE.MatchString(s) {
		return false
	}
	// Block path traversal even if individual chars are allowed.
	return !strings.Contains(s, "..")
}
