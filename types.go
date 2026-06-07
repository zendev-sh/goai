package goai

import (
	"context"
	"encoding/json"
	"fmt"
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

// NewTool builds a Tool from a typed input struct and a typed execute function.
// The JSON Schema is generated from In via SchemaFrom, and the raw JSON
// arguments from the model are unmarshaled into In before execute runs - so
// callers neither hand-write JSON Schema nor unmarshal input themselves.
//
// In is typically a struct whose exported fields carry json and jsonschema tags
// (see SchemaFrom for the supported tags). Use struct{} for a tool that takes no
// parameters.
//
// If the model sends arguments that do not unmarshal into In, execute is not
// called and the error is returned to the model as the tool result. Empty
// arguments leave In at its zero value.
//
//	weather := goai.NewTool("get_weather", "Get the weather for a city",
//		func(ctx context.Context, in struct {
//			City string `json:"city" jsonschema:"description=City name"`
//		}) (string, error) {
//			return forecast(in.City), nil
//		})
func NewTool[In any](name, description string, execute func(ctx context.Context, input In) (string, error)) Tool {
	return Tool{
		Name:        name,
		Description: description,
		InputSchema: SchemaFrom[In](),
		Execute: func(ctx context.Context, raw json.RawMessage) (string, error) {
			var in In
			if len(raw) > 0 {
				if err := json.Unmarshal(raw, &in); err != nil {
					return "", fmt.Errorf("goai: tool %q: invalid input: %w", name, err)
				}
			}
			return execute(ctx, in)
		},
	}
}
