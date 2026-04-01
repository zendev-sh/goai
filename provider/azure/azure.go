// Package azure provides an Azure OpenAI language model implementation for GoAI.
//
// Azure delegates to the OpenAI provider with a custom HTTP transport that
// rewrites URLs to Azure's deployment-based format and injects Azure auth.
// This matches Vercel AI SDK's pattern where @ai-sdk/azure creates an
// OpenAIChatLanguageModel with a custom URL builder.
//
// Usage:
//
//	model := azure.Chat("gpt-4o",
//		azure.WithAPIKey("your-api-key"),
//		azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	)
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package azure

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"net/http"
	"os"
	"strings"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/anthropic"
	"github.com/zendev-sh/goai/provider/openai"
)

var (
	_ provider.LanguageModel = (*chatCompletionsModel)(nil)
	_ provider.CapableModel  = (*chatCompletionsModel)(nil)
)

const defaultAPIVersion = "v1"

// Option configures the Azure OpenAI provider.
type Option func(*options)

type options struct {
	apiKey                 string
	tokenSource            provider.TokenSource
	endpoint               string
	headers                map[string]string
	httpClient             *http.Client
	apiVersion             string
	useDeploymentBasedURLs bool
}

// WithAPIKey sets the Azure API key.
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.apiKey = key
	}
}

// WithTokenSource sets a dynamic token source (e.g. Managed Identity).
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) {
		o.tokenSource = ts
	}
}

// WithEndpoint sets the Azure OpenAI endpoint URL.
func WithEndpoint(endpoint string) Option {
	return func(o *options) {
		o.endpoint = endpoint
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

// WithAPIVersion sets the Azure OpenAI API version used in the api-version
// query parameter. Defaults to "v1". Matches Vercel AI SDK's apiVersion option.
func WithAPIVersion(v string) Option {
	return func(o *options) {
		o.apiVersion = v
	}
}

// WithDeploymentBasedURLs enables legacy deployment-based URL format:
//
//	{endpoint}/openai/deployments/{model}{path}?api-version={version}
//
// instead of the newer format:
//
//	{endpoint}/openai/v1{path}?api-version={version}
//
// This matches Vercel AI SDK's useDeploymentBasedUrls option.
// When enabled, ALL requests (including Responses API) use deployment-based URLs.
func WithDeploymentBasedURLs(enabled bool) Option {
	return func(o *options) {
		o.useDeploymentBasedURLs = enabled
	}
}

// resolveOptions applies defaults and environment variable fallbacks.
func resolveOptions(o *options) {
	// Resolve endpoint from env if not set.
	if o.endpoint == "" {
		o.endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if o.endpoint == "" {
		if r := os.Getenv("AZURE_RESOURCE_NAME"); r != "" {
			o.endpoint = fmt.Sprintf("https://%s.openai.azure.com", r)
		}
	}
	// Resolve API key from env if not set.
	if o.apiKey == "" && o.tokenSource == nil {
		o.apiKey = cmp.Or(os.Getenv("AZURE_OPENAI_API_KEY"), os.Getenv("AZURE_API_KEY"))
	}
	// Default API version.
	o.apiVersion = cmp.Or(o.apiVersion, defaultAPIVersion)
}

// buildHTTPClient creates an HTTP client with the Azure transport that rewrites
// URLs and injects auth. Shared between Chat() and Image().
func buildHTTPClient(o *options, modelID string) *http.Client {
	baseTransport := http.DefaultTransport
	if o.httpClient != nil && o.httpClient.Transport != nil {
		baseTransport = o.httpClient.Transport
	}

	azureTransport := &azureRoundTripper{
		base:                   baseTransport,
		endpoint:               strings.TrimRight(o.endpoint, "/"),
		apiKey:                 o.apiKey,
		tokenSource:            o.tokenSource,
		headers:                o.headers,
		deployID:               strings.ReplaceAll(modelID, "/", "-"),
		apiVersion:             o.apiVersion,
		useDeploymentBasedURLs: o.useDeploymentBasedURLs,
	}

	return &http.Client{Transport: azureTransport}
}

// Chat creates an Azure language model for the given model/deployment ID.
//
// For OpenAI models (GPT, o-series): delegates to openai.Chat() with a custom
// HTTP transport that rewrites URLs to Azure's format and injects Azure auth.
//
// For Claude models: delegates to anthropic.Chat() using Azure's Anthropic
// endpoint (https://{resource}.services.ai.azure.com/anthropic). Azure hosts
// Claude via a separate API that speaks the Anthropic Messages protocol, not
// OpenAI Chat Completions.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	resolveOptions(&o)

	// Claude models on Azure use the Anthropic Messages API at a different
	// endpoint than OpenAI models. Detect and delegate accordingly.
	if strings.Contains(strings.ToLower(modelID), "claude") {
		return buildAnthropicModel(&o, modelID)
	}

	// Non-OpenAI models on Azure (Cohere, DeepSeek, Grok, Llama, Mistral, Phi,
	// Kimi, etc.) use the Azure AI Services endpoint, not the OpenAI endpoint.
	// Route them to https://{resource}.services.ai.azure.com/models/chat/completions.
	if !isOpenAIModel(modelID) {
		return buildAIServicesModel(&o, modelID)
	}

	httpClient := buildHTTPClient(&o, modelID)

	// Delegate to openai.Chat which handles Chat Completions vs Responses API routing.
	// Provide a dummy API key so the OpenAI provider doesn't error on auth --
	// azureRoundTripper replaces the Authorization header with Azure auth.
	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		// Use a placeholder base URL -- azureRoundTripper rewrites it.
		openai.WithBaseURL("https://azure-placeholder"),
	}

	return openai.Chat(modelID, openaiOpts...)
}

// isOpenAIModel returns true if the model ID is an OpenAI-native model
// (GPT, o-series, codex, chatgpt) that supports the Responses API.
func isOpenAIModel(modelID string) bool {
	id := strings.ToLower(modelID)
	// Strip provider prefix (e.g., "openai/gpt-4o" → "gpt-4o").
	if idx := strings.LastIndex(id, "/"); idx >= 0 {
		id = id[idx+1:]
	}
	switch {
	case strings.HasPrefix(id, "gpt-"):
		return true
	case len(id) >= 2 && id[0] == 'o' && id[1] >= '0' && id[1] <= '9':
		return true // o1, o3, o4, etc.
	case strings.HasPrefix(id, "codex"):
		return true
	case strings.HasPrefix(id, "chatgpt"):
		return true
	case strings.HasPrefix(id, "computer-use"):
		return true
	default:
		return false
	}
}

// chatCompletionsModel wraps a LanguageModel to force Chat Completions API
// by injecting useResponsesAPI=false into ProviderOptions.
type chatCompletionsModel struct {
	inner provider.LanguageModel
}

func (m *chatCompletionsModel) ModelID() string { return m.inner.ModelID() }

func (m *chatCompletionsModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilitiesOf(m.inner)
}

func (m *chatCompletionsModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: azure: WithPromptCaching is not supported and will be ignored\n")
	}
	forceChatCompletions(&params)
	return m.inner.DoGenerate(ctx, params)
}

func (m *chatCompletionsModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: azure: WithPromptCaching is not supported and will be ignored\n")
	}
	forceChatCompletions(&params)
	return m.inner.DoStream(ctx, params)
}

func forceChatCompletions(params *provider.GenerateParams) {
	// Copy the map to avoid mutating the caller's ProviderOptions.
	newOpts := maps.Clone(params.ProviderOptions)
	if newOpts == nil {
		newOpts = make(map[string]any, 1)
	}
	newOpts["useResponsesAPI"] = false
	params.ProviderOptions = newOpts
}

// extractResourceName gets the Azure resource name from env or endpoint URL.
func extractResourceName(o *options) string {
	if r := os.Getenv("AZURE_RESOURCE_NAME"); r != "" {
		return r
	}
	if o.endpoint != "" {
		ep := strings.TrimPrefix(o.endpoint, "https://")
		ep = strings.TrimPrefix(ep, "http://")
		if idx := strings.Index(ep, "."); idx > 0 {
			return ep[:idx]
		}
	}
	return ""
}

// buildAnthropicModel creates an Anthropic provider pointing at Azure's
// Anthropic endpoint: https://{resource}.services.ai.azure.com/anthropic
func buildAnthropicModel(o *options, modelID string) provider.LanguageModel {
	resourceName := extractResourceName(o)
	anthropicEndpoint := fmt.Sprintf("https://%s.services.ai.azure.com/anthropic", resourceName)

	anthropicOpts := []anthropic.Option{anthropic.WithBaseURL(anthropicEndpoint)}
	if o.apiKey != "" {
		anthropicOpts = append(anthropicOpts, anthropic.WithAPIKey(o.apiKey))
	}
	if o.tokenSource != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithTokenSource(o.tokenSource))
	}
	if o.httpClient != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithHTTPClient(o.httpClient))
	}
	if len(o.headers) > 0 {
		anthropicOpts = append(anthropicOpts, anthropic.WithHeaders(o.headers))
	}
	return anthropic.Chat(modelID, anthropicOpts...)
}

// defaultAIServicesAPIVersion is used for Azure AI Services (non-OpenAI, non-Anthropic models).
const defaultAIServicesAPIVersion = "2024-05-01-preview"

// buildAIServicesModel creates an OpenAI-compat model pointing at Azure AI Services
// endpoint: https://{resource}.services.ai.azure.com/models
// This endpoint hosts non-OpenAI, non-Anthropic models (DeepSeek, Llama, Phi, Grok,
// Cohere, Mistral, Kimi, etc.) using standard OpenAI Chat Completions format.
func buildAIServicesModel(o *options, modelID string) provider.LanguageModel {
	resourceName := extractResourceName(o)
	baseURL := fmt.Sprintf("https://%s.services.ai.azure.com/models", resourceName)

	baseTransport := http.DefaultTransport
	if o.httpClient != nil && o.httpClient.Transport != nil {
		baseTransport = o.httpClient.Transport
	}

	transport := &aiServicesRoundTripper{
		base:        baseTransport,
		apiKey:      o.apiKey,
		tokenSource: o.tokenSource,
		headers:     o.headers,
		apiVersion:  defaultAIServicesAPIVersion,
	}

	httpClient := &http.Client{Transport: transport}

	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL(baseURL),
	}

	model := openai.Chat(modelID, openaiOpts...)
	return &chatCompletionsModel{inner: model}
}

// aiServicesRoundTripper injects Azure auth headers and api-version query parameter
// for the Azure AI Services endpoint. Unlike azureRoundTripper, it does NOT rewrite
// the URL path -- the OpenAI provider already builds the correct /chat/completions path.
type aiServicesRoundTripper struct {
	base        http.RoundTripper
	apiKey      string
	tokenSource provider.TokenSource
	headers     map[string]string
	apiVersion  string
}

func (t *aiServicesRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	newReq := req.Clone(req.Context())

	// Add api-version query parameter.
	q := newReq.URL.Query()
	q.Set("api-version", t.apiVersion)
	newReq.URL.RawQuery = q.Encode()

	// Replace OpenAI auth with Azure auth.
	newReq.Header.Del("Authorization")
	if t.apiKey != "" {
		newReq.Header.Set("api-key", t.apiKey)
	} else if t.tokenSource != nil {
		token, err := t.tokenSource.Token(req.Context())
		if err != nil {
			return nil, fmt.Errorf("azure: resolving auth token: %w", err)
		}
		newReq.Header.Set("Authorization", "Bearer "+token)
	}

	// Inject custom headers.
	for k, v := range t.headers {
		newReq.Header.Set(k, v)
	}

	return t.base.RoundTrip(newReq)
}

// Image creates an Azure OpenAI image model (DALL-E) for the given deployment ID.
//
// Internally delegates to openai.Image() with a custom HTTP transport that
// rewrites URLs to Azure's format and injects Azure auth. This matches
// Vercel AI SDK's azure.image() which delegates to OpenAIImageModel.
func Image(modelID string, opts ...Option) provider.ImageModel {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	resolveOptions(&o)

	httpClient := buildHTTPClient(&o, modelID)

	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL("https://azure-placeholder"),
	}

	return openai.Image(modelID, openaiOpts...)
}

// azureRoundTripper rewrites OpenAI API URLs to Azure URLs and injects
// Azure authentication headers.
//
// Default (useDeploymentBasedURLs=false, matches Vercel default):
//
//	{endpoint}/openai/v1{path}?api-version={version}
//
// Legacy (useDeploymentBasedURLs=true):
//
//	{endpoint}/openai/deployments/{model}{path}?api-version={version}
type azureRoundTripper struct {
	base                   http.RoundTripper
	endpoint               string
	apiKey                 string
	tokenSource            provider.TokenSource
	headers                map[string]string
	deployID               string
	apiVersion             string
	useDeploymentBasedURLs bool
}

func (t *azureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// Extract the API path suffix from the OpenAI URL.
	path := req.URL.Path
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		path = "/chat/completions"
	case strings.HasSuffix(path, "/responses"):
		path = "/responses"
	case strings.HasSuffix(path, "/images/generations"):
		path = "/images/generations"
	}

	var azureURL string
	if t.useDeploymentBasedURLs {
		// Legacy deployment-based format for ALL endpoints.
		azureURL = fmt.Sprintf("%s/openai/deployments/%s%s?api-version=%s",
			t.endpoint, t.deployID, path, t.apiVersion)
	} else {
		// Default: newer v1 format (model in body).
		azureURL = fmt.Sprintf("%s/openai/v1%s?api-version=%s",
			t.endpoint, path, t.apiVersion)
	}

	// Clone the request with the new URL.
	newReq := req.Clone(req.Context())
	parsed, err := req.URL.Parse(azureURL)
	if err != nil {
		return nil, fmt.Errorf("azure: invalid URL: %w", err)
	}
	newReq.URL = parsed
	newReq.Host = parsed.Host

	// Replace OpenAI auth with Azure auth.
	// API key uses Azure's api-key header; TokenSource uses Bearer (Managed Identity, etc.).
	newReq.Header.Del("Authorization")
	if t.apiKey != "" {
		newReq.Header.Set("api-key", t.apiKey)
	} else if t.tokenSource != nil {
		token, tokenErr := t.tokenSource.Token(req.Context())
		if tokenErr != nil {
			return nil, fmt.Errorf("azure: resolving auth token: %w", tokenErr)
		}
		newReq.Header.Set("Authorization", "Bearer "+token)
	}

	// Inject Azure-specific headers.
	for k, v := range t.headers {
		newReq.Header.Set(k, v)
	}

	return t.base.RoundTrip(newReq)
}
