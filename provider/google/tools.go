package google

import (
	"strings"

	"github.com/zendev-sh/goai/provider"
)

// Tools provides factory functions for Google provider-defined tools.
// These tools use Gemini's built-in capabilities (Google Search grounding,
// URL context, code execution). Requires Gemini 2.0+.
// Matches Vercel AI SDK's google.tools.
var Tools = struct {
	// GoogleSearch enables grounding with Google Search.
	// The model decides when to search based on the prompt.
	// Returns sources via groundingMetadata in response.
	GoogleSearch func(opts ...GoogleSearchOption) provider.ToolDefinition

	// URLContext enables URL context tool that gives Gemini access to web content.
	// The model uses URLs from the prompt to fetch and process content.
	// Requires Gemini 2.0+.
	URLContext func() provider.ToolDefinition

	// CodeExecution enables the model to generate and run Python code.
	// The model can write code, execute it in a sandboxed environment, and
	// use the output to formulate its response.
	// Requires Gemini 2.0+.
	CodeExecution func() provider.ToolDefinition
}{
	GoogleSearch:  googleSearchTool,
	URLContext:    urlContextTool,
	CodeExecution: codeExecutionTool,
}

// ---------------------------------------------------------------------------
// GoogleSearch
// ---------------------------------------------------------------------------

// GoogleSearchOption configures the Google Search grounding tool.
type GoogleSearchOption func(*googleSearchConfig)

type googleSearchConfig struct {
	// SearchTypes controls which search types to use.
	WebSearch   bool
	ImageSearch bool
	// TimeRangeFilter restricts results to a time range.
	StartTime string // RFC3339 format
	EndTime   string // RFC3339 format
}

// WithWebSearch enables web search results.
func WithWebSearch() GoogleSearchOption {
	return func(c *googleSearchConfig) { c.WebSearch = true }
}

// WithImageSearch enables image search results.
func WithImageSearch() GoogleSearchOption {
	return func(c *googleSearchConfig) { c.ImageSearch = true }
}

// WithTimeRange restricts search results to a specific time range (RFC3339 format).
func WithTimeRange(startTime, endTime string) GoogleSearchOption {
	return func(c *googleSearchConfig) {
		c.StartTime = startTime
		c.EndTime = endTime
	}
}

func googleSearchTool(opts ...GoogleSearchOption) provider.ToolDefinition {
	cfg := &googleSearchConfig{}
	for _, o := range opts {
		o(cfg)
	}

	providerOpts := map[string]any{}

	if cfg.WebSearch || cfg.ImageSearch {
		searchTypes := map[string]any{}
		if cfg.WebSearch {
			searchTypes["webSearch"] = map[string]any{}
		}
		if cfg.ImageSearch {
			searchTypes["imageSearch"] = map[string]any{}
		}
		providerOpts["searchTypes"] = searchTypes
	}

	if cfg.StartTime != "" && cfg.EndTime != "" {
		providerOpts["timeRangeFilter"] = map[string]any{
			"startTime": cfg.StartTime,
			"endTime":   cfg.EndTime,
		}
	}

	return provider.ToolDefinition{
		Name:                   "google_search",
		ProviderDefinedType:    "google.google_search",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// URLContext
// ---------------------------------------------------------------------------

func urlContextTool() provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "url_context",
		ProviderDefinedType: "google.url_context",
	}
}

// ---------------------------------------------------------------------------
// CodeExecution
// ---------------------------------------------------------------------------

func codeExecutionTool() provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "code_execution",
		ProviderDefinedType: "google.code_execution",
	}
}

// ---------------------------------------------------------------------------
// API conversion helpers
// ---------------------------------------------------------------------------

// googleProviderTool maps a ProviderDefinedType to the Gemini API tool format.
// Gemini uses camelCase keys: {"googleSearch": {...}}, {"urlContext": {}}, {"codeExecution": {}}.
func googleProviderTool(t provider.ToolDefinition) map[string]any {
	// Map "google.google_search" -> "googleSearch", "google.url_context" -> "urlContext", etc.
	apiKey := t.ProviderDefinedType
	apiKey = strings.TrimPrefix(apiKey, "google.")
	// Convert snake_case to camelCase: "google_search" -> "googleSearch"
	apiKey = snakeToCamel(apiKey)

	opts := map[string]any{}
	for k, v := range t.ProviderDefinedOptions {
		opts[k] = v
	}
	return map[string]any{apiKey: opts}
}

func snakeToCamel(s string) string {
	parts := strings.Split(s, "_")
	for i := 1; i < len(parts); i++ {
		if len(parts[i]) > 0 {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}
