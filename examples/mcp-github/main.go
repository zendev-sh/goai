//go:build ignore

// Example: GitHub MCP server integration.
//
// Connects to the official GitHub MCP server via stdio to discover and
// call GitHub tools. Requires Node.js (npx) and a GITHUB_TOKEN.
//
// Usage:
//
//	export GITHUB_TOKEN=ghp_...
//	go run ./examples/mcp-github/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	token := os.Getenv("GITHUB_TOKEN")
	if token == "" {
		log.Fatal("GITHUB_TOKEN environment variable is required")
	}

	ctx := context.Background()

	// Launch the GitHub MCP server via npx. The server reads GITHUB_TOKEN
	// from its environment to authenticate GitHub API calls.
	transport := mcp.NewStdioTransport("npx", []string{
		"-y", "@modelcontextprotocol/server-github",
	}, mcp.WithStdioEnv(map[string]string{
		"GITHUB_TOKEN": token,
	}))

	client := mcp.NewClient("goai-mcp-github", "1.0.0", mcp.WithTransport(transport))
	fmt.Println("Connecting to GitHub MCP server (this may take a moment on first run)...")
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("Connected to %s v%s\n\n", info.Name, info.Version)

	// --- List all available tools ---
	fmt.Println("=== Available GitHub Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	total := len(toolsResult.Tools)
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %-40s %s\n", tool.Name, truncate(tool.Description, 60))
	}
	// Paginate if needed.
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		total += len(toolsResult.Tools)
		for _, tool := range toolsResult.Tools {
			fmt.Printf("  %-40s %s\n", tool.Name, truncate(tool.Description, 60))
		}
	}
	fmt.Printf("\nTotal: %d tools\n", total)

	// --- Call a tool: search for Go repositories ---
	fmt.Println("\n=== Search: top Go repositories ===")
	result, err := client.CallTool(ctx, "search_repositories", map[string]any{
		"query":   "language:go stars:>10000",
		"perPage": 5,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(mcp.FormatContent(result.Content, result.IsError))
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
