package anthropic

import (
	"encoding/json"

	"github.com/zendev-sh/goai/provider"
)

// Tools provides factory functions for Anthropic provider-defined tools.
// These tools use Anthropic's built-in tool types (computer use, bash, text editor,
// web search, web fetch, code execution) with predefined schemas.
// Matches Vercel AI SDK's anthropic.tools.
//
// Default versions are the latest stable for each tool.
// Use versioned constructors (e.g. Tools.Computer_20251124) for specific versions.
var Tools = struct {
	// Computer creates a computer use tool definition (version 20250124).
	// Supports screenshot, mouse/keyboard control for autonomous desktop interaction.
	Computer func(opts ComputerToolOptions) provider.ToolDefinition

	// Computer_20251124 creates a computer use tool with zoom support (Opus 4.5+).
	Computer_20251124 func(opts Computer20251124Options) provider.ToolDefinition

	// Bash creates a bash tool definition (version 20250124).
	Bash func() provider.ToolDefinition

	// TextEditor creates a text editor tool definition (version 20250429).
	// Does not support "undo_edit" command.
	TextEditor func() provider.ToolDefinition

	// TextEditor_20250728 creates a text editor tool with optional maxCharacters.
	// Supported models: Claude Sonnet 4, Opus 4, Opus 4.1.
	TextEditor_20250728 func(opts ...TextEditorOption) provider.ToolDefinition

	// WebSearch creates a web search tool definition (version 20250305).
	// Claude decides when to search based on the prompt. Uses Brave Search.
	WebSearch func(opts ...WebSearchOption) provider.ToolDefinition

	// WebSearch_20260209 creates a web search tool (version 20260209).
	// Requires beta header: code-execution-web-tools-2026-02-09.
	WebSearch_20260209 func(opts ...WebSearchOption) provider.ToolDefinition

	// WebFetch creates a web fetch tool definition (version 20260209).
	// Gives Claude direct access to real-time web content via URL fetching.
	WebFetch func(opts ...WebFetchOption) provider.ToolDefinition

	// CodeExecution creates a code execution tool definition (version 20260120).
	// Does not require a beta header. Recommended version.
	// Supported models: Claude Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5.
	CodeExecution func() provider.ToolDefinition

	// CodeExecution_20250825 creates a code execution tool (version 20250825).
	// Requires beta header: code-execution-2025-08-25.
	CodeExecution_20250825 func() provider.ToolDefinition
}{
	Computer: func(opts ComputerToolOptions) provider.ToolDefinition {
		return computerTool(opts, "computer_20250124")
	},
	Computer_20251124: computer20251124Tool,
	Bash: func() provider.ToolDefinition {
		return bashTool("bash_20250124")
	},
	TextEditor: func() provider.ToolDefinition {
		return textEditorTool("text_editor_20250429")
	},
	TextEditor_20250728: textEditor20250728Tool,
	WebSearch:           webSearchTool,
	WebSearch_20260209:  webSearch20260209Tool,
	WebFetch:            webFetchTool,
	CodeExecution: func() provider.ToolDefinition {
		return codeExecutionTool("code_execution_20260120")
	},
	CodeExecution_20250825: func() provider.ToolDefinition {
		return codeExecutionTool("code_execution_20250825")
	},
}

// ---------------------------------------------------------------------------
// Computer
// ---------------------------------------------------------------------------

// ComputerToolOptions configures the computer use tool (20250124).
type ComputerToolOptions struct {
	// DisplayWidthPx is the width of the computer display in pixels (required).
	DisplayWidthPx int
	// DisplayHeightPx is the height of the computer display in pixels (required).
	DisplayHeightPx int
	// DisplayNumber is the X11 display number (optional).
	DisplayNumber int
}

func computerTool(opts ComputerToolOptions, version string) provider.ToolDefinition {
	providerOpts := map[string]any{
		"display_width_px":  opts.DisplayWidthPx,
		"display_height_px": opts.DisplayHeightPx,
	}
	if opts.DisplayNumber > 0 {
		providerOpts["display_number"] = opts.DisplayNumber
	}
	return provider.ToolDefinition{
		Name:                   "computer",
		ProviderDefinedType:    version,
		ProviderDefinedOptions: providerOpts,
	}
}

// Computer20251124Options configures the computer use tool (20251124) with zoom support.
type Computer20251124Options struct {
	// DisplayWidthPx is the width of the computer display in pixels (required).
	DisplayWidthPx int
	// DisplayHeightPx is the height of the computer display in pixels (required).
	DisplayHeightPx int
	// DisplayNumber is the X11 display number (optional).
	DisplayNumber int
	// EnableZoom enables the zoom action for inspecting specific screen regions.
	EnableZoom bool
}

func computer20251124Tool(opts Computer20251124Options) provider.ToolDefinition {
	providerOpts := map[string]any{
		"display_width_px":  opts.DisplayWidthPx,
		"display_height_px": opts.DisplayHeightPx,
	}
	if opts.DisplayNumber > 0 {
		providerOpts["display_number"] = opts.DisplayNumber
	}
	if opts.EnableZoom {
		providerOpts["enable_zoom"] = true
	}
	return provider.ToolDefinition{
		Name:                   "computer",
		ProviderDefinedType:    "computer_20251124",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// Bash
// ---------------------------------------------------------------------------

func bashTool(version string) provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "bash",
		ProviderDefinedType: version,
	}
}

// ---------------------------------------------------------------------------
// TextEditor
// ---------------------------------------------------------------------------

func textEditorTool(version string) provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "str_replace_based_edit_tool",
		ProviderDefinedType: version,
	}
}

// TextEditorOption configures the text editor tool (20250728+).
type TextEditorOption func(*textEditorConfig)

type textEditorConfig struct {
	MaxCharacters int
}

// WithMaxCharacters sets the maximum number of characters to view in a file.
func WithMaxCharacters(n int) TextEditorOption {
	return func(c *textEditorConfig) { c.MaxCharacters = n }
}

func textEditor20250728Tool(opts ...TextEditorOption) provider.ToolDefinition {
	cfg := &textEditorConfig{}
	for _, o := range opts {
		o(cfg)
	}

	providerOpts := map[string]any{}
	if cfg.MaxCharacters > 0 {
		providerOpts["max_characters"] = cfg.MaxCharacters
	}

	return provider.ToolDefinition{
		Name:                   "str_replace_based_edit_tool",
		ProviderDefinedType:    "text_editor_20250728",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// WebSearch
// ---------------------------------------------------------------------------

// WebSearchOption configures the Anthropic web search tool.
type WebSearchOption func(*webSearchConfig)

type webSearchConfig struct {
	MaxUses        int                 // max number of searches per turn
	AllowedDomains []string            // restrict search to these domains
	BlockedDomains []string            // exclude these domains from search
	UserLocation   *WebSearchLocation  // optional user location
}

// WebSearchLocation provides geographically relevant search results.
type WebSearchLocation struct {
	Type     string // always "approximate"
	City     string
	Region   string
	Country  string
	Timezone string // IANA timezone ID
}

// WithMaxUses limits the number of web searches per turn.
func WithMaxUses(n int) WebSearchOption {
	return func(c *webSearchConfig) { c.MaxUses = n }
}

// WithAllowedDomains restricts search results to these domains.
func WithAllowedDomains(domains ...string) WebSearchOption {
	return func(c *webSearchConfig) { c.AllowedDomains = domains }
}

// WithBlockedDomains excludes these domains from search results.
func WithBlockedDomains(domains ...string) WebSearchOption {
	return func(c *webSearchConfig) { c.BlockedDomains = domains }
}

// WithWebSearchUserLocation provides user location for geographically relevant results.
func WithWebSearchUserLocation(loc WebSearchLocation) WebSearchOption {
	return func(c *webSearchConfig) { c.UserLocation = &loc }
}

func buildWebSearchOptions(cfg *webSearchConfig) map[string]any {
	providerOpts := map[string]any{}
	if cfg.MaxUses > 0 {
		providerOpts["max_uses"] = cfg.MaxUses
	}
	if len(cfg.AllowedDomains) > 0 {
		providerOpts["allowed_domains"] = cfg.AllowedDomains
	}
	if len(cfg.BlockedDomains) > 0 {
		providerOpts["blocked_domains"] = cfg.BlockedDomains
	}
	if cfg.UserLocation != nil {
		loc := map[string]any{"type": "approximate"}
		if cfg.UserLocation.City != "" {
			loc["city"] = cfg.UserLocation.City
		}
		if cfg.UserLocation.Region != "" {
			loc["region"] = cfg.UserLocation.Region
		}
		if cfg.UserLocation.Country != "" {
			loc["country"] = cfg.UserLocation.Country
		}
		if cfg.UserLocation.Timezone != "" {
			loc["timezone"] = cfg.UserLocation.Timezone
		}
		providerOpts["user_location"] = loc
	}
	return providerOpts
}

func webSearchTool(opts ...WebSearchOption) provider.ToolDefinition {
	cfg := &webSearchConfig{}
	for _, o := range opts {
		o(cfg)
	}

	return provider.ToolDefinition{
		Name:                   "web_search",
		ProviderDefinedType:    "web_search_20250305",
		ProviderDefinedOptions: buildWebSearchOptions(cfg),
	}
}

func webSearch20260209Tool(opts ...WebSearchOption) provider.ToolDefinition {
	cfg := &webSearchConfig{}
	for _, o := range opts {
		o(cfg)
	}

	return provider.ToolDefinition{
		Name:                   "web_search",
		ProviderDefinedType:    "web_search_20260209",
		ProviderDefinedOptions: buildWebSearchOptions(cfg),
	}
}

// ---------------------------------------------------------------------------
// WebFetch
// ---------------------------------------------------------------------------

// WebFetchOption configures the Anthropic web fetch tool.
type WebFetchOption func(*webFetchConfig)

type webFetchConfig struct {
	MaxUses          int
	AllowedDomains   []string
	BlockedDomains   []string
	Citations        *WebFetchCitations
	MaxContentTokens int
}

// WebFetchCitations controls citation behavior for web fetch results.
type WebFetchCitations struct {
	Enabled bool
}

// WithWebFetchMaxUses limits the number of web fetches per turn.
func WithWebFetchMaxUses(n int) WebFetchOption {
	return func(c *webFetchConfig) { c.MaxUses = n }
}

// WithWebFetchAllowedDomains restricts fetching to these domains.
func WithWebFetchAllowedDomains(domains ...string) WebFetchOption {
	return func(c *webFetchConfig) { c.AllowedDomains = domains }
}

// WithWebFetchBlockedDomains excludes these domains from fetching.
func WithWebFetchBlockedDomains(domains ...string) WebFetchOption {
	return func(c *webFetchConfig) { c.BlockedDomains = domains }
}

// WithCitations enables citations for fetched content.
func WithCitations(enabled bool) WebFetchOption {
	return func(c *webFetchConfig) { c.Citations = &WebFetchCitations{Enabled: enabled} }
}

// WithMaxContentTokens limits the amount of content included in context.
func WithMaxContentTokens(n int) WebFetchOption {
	return func(c *webFetchConfig) { c.MaxContentTokens = n }
}

func webFetchTool(opts ...WebFetchOption) provider.ToolDefinition {
	cfg := &webFetchConfig{}
	for _, o := range opts {
		o(cfg)
	}

	providerOpts := map[string]any{}
	if cfg.MaxUses > 0 {
		providerOpts["max_uses"] = cfg.MaxUses
	}
	if len(cfg.AllowedDomains) > 0 {
		providerOpts["allowed_domains"] = cfg.AllowedDomains
	}
	if len(cfg.BlockedDomains) > 0 {
		providerOpts["blocked_domains"] = cfg.BlockedDomains
	}
	if cfg.Citations != nil {
		providerOpts["citations"] = map[string]any{"enabled": cfg.Citations.Enabled}
	}
	if cfg.MaxContentTokens > 0 {
		providerOpts["max_content_tokens"] = cfg.MaxContentTokens
	}

	return provider.ToolDefinition{
		Name:                   "web_fetch",
		ProviderDefinedType:    "web_fetch_20260209",
		ProviderDefinedOptions: providerOpts,
	}
}

// ---------------------------------------------------------------------------
// CodeExecution
// ---------------------------------------------------------------------------

func codeExecutionTool(version string) provider.ToolDefinition {
	return provider.ToolDefinition{
		Name:                "code_execution",
		ProviderDefinedType: version,
	}
}

// ---------------------------------------------------------------------------
// Beta header helpers
// ---------------------------------------------------------------------------

// betaForTool returns the anthropic-beta header value required for a
// provider-defined tool type. Returns empty string if no beta is needed.
// Keep in sync with provider/bedrock/bedrock.go toolBetaForType.
func betaForTool(toolType string) string {
	switch toolType {
	case "computer_20241022", "bash_20241022", "text_editor_20241022":
		return "computer-use-2024-10-22"
	case "computer_20250124", "bash_20250124", "text_editor_20250124", "text_editor_20250429":
		return "computer-use-2025-01-24"
	case "computer_20251124":
		return "computer-use-2025-11-24"
	case "text_editor_20250728":
		return "" // no beta needed
	case "code_execution_20250825":
		return "code-execution-2025-08-25"
	case "code_execution_20260120":
		return "" // no beta needed
	case "web_search_20250305":
		return "web-search-2025-03-05"
	case "web_search_20260209", "web_fetch_20260209":
		return "code-execution-web-tools-2026-02-09"
	default:
		return ""
	}
}

// collectToolBetas extracts unique beta header values from provider-defined tools.
func collectToolBetas(tools []provider.ToolDefinition) []string {
	seen := make(map[string]bool)
	var betas []string
	for _, t := range tools {
		if t.ProviderDefinedType == "" {
			continue
		}
		beta := betaForTool(t.ProviderDefinedType)
		if beta != "" && !seen[beta] {
			seen[beta] = true
			betas = append(betas, beta)
		}
	}
	return betas
}

// convertToolToAPI converts a ToolDefinition into the Anthropic API tool format.
// For provider-defined tools, it uses the special type instead of "custom".
func convertToolToAPI(t provider.ToolDefinition) map[string]any {
	if t.ProviderDefinedType != "" {
		tool := map[string]any{
			"type": t.ProviderDefinedType,
			"name": t.Name,
		}
		// Merge provider-defined options as top-level fields.
		for k, v := range t.ProviderDefinedOptions {
			tool[k] = v
		}
		return tool
	}

	// Regular (custom) tool.
	tool := map[string]any{
		"name":        t.Name,
		"description": t.Description,
	}
	if len(t.InputSchema) > 0 {
		var schema any
		if err := json.Unmarshal(t.InputSchema, &schema); err == nil {
			tool["input_schema"] = schema
		}
	}
	return tool
}
