//go:build ignore

// Example: filesystem MCP server integration.
//
// Connects to the official MCP filesystem server via stdio to browse
// and read files in a specified directory. Requires Node.js (npx).
//
// Usage:
//
//	go run ./examples/mcp-filesystem/main.go /tmp
//	go run ./examples/mcp-filesystem/main.go .
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run ./examples/mcp-filesystem/main.go <directory>")
	}
	dir := os.Args[1]
	ctx := context.Background()

	// Launch the filesystem MCP server. It exposes tools for reading,
	// writing, listing, and searching files within the allowed directory.
	transport := mcp.NewStdioTransport("npx", []string{
		"-y", "@modelcontextprotocol/server-filesystem", dir,
	})

	client := mcp.NewClient("goai-mcp-fs", "1.0.0", mcp.WithTransport(transport))
	fmt.Printf("Connecting to filesystem server for %s ...\n", dir)
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("Connected to %s v%s\n\n", info.Name, info.Version)

	// List available tools (with pagination).
	fmt.Println("=== Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
	}
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		for _, tool := range toolsResult.Tools {
			fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
		}
	}

	// List the directory.
	fmt.Printf("\n=== list_directory(%s) ===\n", dir)
	result, err := client.CallTool(ctx, "list_directory", map[string]any{
		"path": dir,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(mcp.FormatContent(result.Content, result.IsError))
}
