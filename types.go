package goai

import (
	"context"
	"encoding/json"
)

// Tool defines a tool that can be called by the model during generation.
// Unlike provider.ToolDefinition (wire-level schema), Tool includes an
// Execute function that GoAI's auto tool loop invokes.
type Tool struct {
	// Name is the tool's identifier.
	Name string

	// Description explains what the tool does.
	Description string

	// InputSchema is the JSON Schema for the tool's input parameters.
	InputSchema json.RawMessage

	// ProviderDefinedType, when non-empty, marks this as a provider-defined tool
	// (e.g. "computer_20250124", "bash_20250124"). Providers emit the correct
	// API type instead of "custom".
	ProviderDefinedType string

	// ProviderDefinedOptions holds provider-specific tool configuration
	// (e.g. displayWidthPx for computer use).
	ProviderDefinedOptions map[string]any

	// Execute runs the tool with the given JSON input and returns the result text.
	// Both the return value and error string are forwarded to the model as a tool
	// result message. Do not include sensitive data (credentials, internal paths)
	// in error messages as they will be sent to the LLM provider's API.
	Execute func(ctx context.Context, input json.RawMessage) (string, error)
}
