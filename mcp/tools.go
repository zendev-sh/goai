package mcp

import (
	"context"
	"encoding/json"
	"strings"

	goai "github.com/zendev-sh/goai"
)

// ConvertTools converts MCP tools to GoAI tools for use with GenerateText
// and StreamText. Each returned tool's Execute function calls back to the
// MCP server via the provided client.
func ConvertTools(client *Client, mcpTools []Tool) []goai.Tool {
	tools := make([]goai.Tool, 0, len(mcpTools))
	for _, mcpTool := range mcpTools {
		tool := mcpTool // capture by value for closure
		tools = append(tools, goai.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
			Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
				args := make(map[string]any)
				if len(input) > 0 {
					if err := json.Unmarshal(input, &args); err != nil {
						return "", err
					}
				}
				result, err := client.CallTool(ctx, tool.Name, args)
				if err != nil {
					return "", err
				}
				return FormatContent(result.Content, result.IsError), nil
			},
		})
	}
	return tools
}

// FormatContent converts MCP ContentBlocks into a single string suitable for
// use as a GoAI tool result. Text blocks are extracted and joined with newlines.
// If isError is true, an "Error:" prefix is prepended.
func FormatContent(content []ContentBlock, isError bool) string {
	var parts []string
	if isError {
		parts = append(parts, "Error:")
	}
	for _, block := range content {
		if tc, ok := ParseTextContent(block); ok {
			parts = append(parts, tc.Text)
		}
	}
	return strings.Join(parts, "\n")
}
