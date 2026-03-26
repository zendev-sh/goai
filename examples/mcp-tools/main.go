//go:build ignore

// Example: MCP tools with GoAI LLM integration.
//
// Connects to an MCP testserver, converts its tools to GoAI tools, and passes
// them to GenerateText so the LLM can call MCP tools in an agent loop.
//
// Requires a GOOGLE_API_KEY (Gemini) for the LLM. The MCP servers are local
// (no external dependencies beyond the API key).
//
// Usage:
//
//	export GOOGLE_API_KEY=...
//	go run ./examples/mcp-tools
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/mcp"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY environment variable is required")
	}

	ctx := context.Background()
	model := google.Chat("gemini-2.0-flash", google.WithAPIKey(apiKey))

	// Connect to the testserver via stdio and discover its tools.
	// In a real application, you might connect to multiple servers
	// (filesystem, database, API) and aggregate their tools.
	transport := mcp.NewStdioTransport("go", []string{"run", "./mcp/testserver", "--mode=stdio"})
	client := mcp.NewClient("tool-server", "1.0.0", mcp.WithTransport(transport))
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Collect all tools with pagination.
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	allMCPTools := toolsResult.Tools
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		allMCPTools = append(allMCPTools, toolsResult.Tools...)
	}

	// Convert MCP tools to GoAI tools for the LLM.
	allTools := mcp.ConvertTools(client, allMCPTools)
	fmt.Printf("MCP server %q: %d tools\n\n", client.ServerInfo().Name, len(allTools))

	// Use all MCP tools in a single GenerateText call.
	// The model will call echo and add tools as needed to answer the question.
	prompt := "First echo the message 'Hello from GoAI', then calculate 1234 + 5678. Report both results."
	fmt.Printf("Prompt: %s\n\n", prompt)

	result, err := goai.GenerateText(ctx, model,
		goai.WithPrompt(prompt),
		goai.WithTools(allTools...),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("--- Response ---")
	fmt.Println(result.Text)
	fmt.Printf("\nSteps: %d, Tokens: %d in / %d out\n",
		len(result.Steps), result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
