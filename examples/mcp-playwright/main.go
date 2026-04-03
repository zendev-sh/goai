//go:build ignore

// Example: Playwright MCP server integration.
//
// Connects to the Playwright MCP server via stdio to automate browser
// interactions. Requires Node.js (npx).
//
// Usage:
//
//	go run ./examples/mcp-playwright/main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	ctx := context.Background()

	// Launch the Playwright MCP server. It exposes tools for browser
	// automation: navigate, click, fill, screenshot, etc.
	transport := mcp.NewStdioTransport("npx", []string{
		"-y", "@playwright/mcp@latest",
	})

	client := mcp.NewClient("goai-mcp-playwright", "1.0.0", mcp.WithTransport(transport))
	fmt.Println("Connecting to Playwright MCP server (first run downloads browser)...")
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("Connected to %s v%s\n\n", info.Name, info.Version)

	// List available tools.
	fmt.Println("=== Playwright Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	total := len(toolsResult.Tools)
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %-30s %s\n", tool.Name, truncate(tool.Description, 60))
	}
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		total += len(toolsResult.Tools)
		for _, tool := range toolsResult.Tools {
			fmt.Printf("  %-30s %s\n", tool.Name, truncate(tool.Description, 60))
		}
	}
	fmt.Printf("\nTotal: %d tools\n", total)

	// Navigate to a page.
	fmt.Println("\n=== Navigate to example.com ===")
	result, err := client.CallTool(ctx, "browser_navigate", map[string]any{
		"url": "https://example.com",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(mcp.FormatContent(result.Content, result.IsError))

	// Take a snapshot of the page.
	fmt.Println("\n=== Page snapshot ===")
	result, err = client.CallTool(ctx, "browser_snapshot", map[string]any{})
	if err != nil {
		log.Fatal(err)
	}
	output := mcp.FormatContent(result.Content, result.IsError)
	fmt.Println(truncate(output, 500))
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
