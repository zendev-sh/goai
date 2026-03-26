//go:build ignore

// Example: MCP client basics  - no LLM needed.
//
// Connects to the project's built-in testserver via stdio transport and
// exercises every MCP capability: tools, prompts, and resources.
//
// Usage:
//
//	go run ./examples/mcp-local
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	ctx := context.Background()

	// Launch the built-in testserver as a stdio subprocess.
	// "go run" compiles and executes it  - works when run from the module root.
	transport := mcp.NewStdioTransport("go", []string{
		"run", "./mcp/testserver", "--mode=stdio",
	})

	client := mcp.NewClient("goai-mcp-local", "1.0.0", mcp.WithTransport(transport))
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("Connected to %s v%s\n", info.Name, info.Version)
	fmt.Printf("Protocol version: %s\n", client.ProtocolVersion())
	if instr := client.Instructions(); instr != "" {
		fmt.Printf("Instructions: %s\n", instr)
	}

	// --- Ping ---
	if err := client.Ping(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nPing: OK")

	// --- Tools ---
	fmt.Println("\n=== Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
	}

	// Paginate to get remaining tools.
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		for _, tool := range toolsResult.Tools {
			fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
		}
	}

	// Call the echo tool.
	echoResult, err := client.CallTool(ctx, "echo", map[string]any{
		"message": "Hello from GoAI!",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\necho(\"Hello from GoAI!\") = %s\n", mcp.FormatContent(echoResult.Content, echoResult.IsError))

	// Call the add tool.
	addResult, err := client.CallTool(ctx, "add", map[string]any{
		"a": 17.0,
		"b": 25.0,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("add(17, 25) = %s\n", mcp.FormatContent(addResult.Content, addResult.IsError))

	// --- Prompts ---
	fmt.Println("\n=== Prompts ===")
	promptsResult, err := client.ListPrompts(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range promptsResult.Prompts {
		args := ""
		for _, a := range p.Arguments {
			if args != "" {
				args += ", "
			}
			args += a.Name
		}
		fmt.Printf("  %s(%s)  - %s\n", p.Name, args, p.Description)
	}
	for promptsResult.NextCursor != "" {
		promptsResult, err = client.ListPrompts(ctx, &mcp.ListParams{Cursor: promptsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		for _, p := range promptsResult.Prompts {
			fmt.Printf("  %s  - %s\n", p.Name, p.Description)
		}
	}

	// Get a prompt with arguments.
	prompt, err := client.GetPrompt(ctx, "greeting", map[string]string{"name": "Gopher"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\ngreeting(name=\"Gopher\"):\n")
	for _, msg := range prompt.Messages {
		if tc, ok := mcp.ParseTextContent(msg.Content); ok {
			fmt.Printf("  [%s] %s\n", msg.Role, tc.Text)
		}
	}

	// --- Resources ---
	fmt.Println("\n=== Resources ===")
	resourcesResult, err := client.ListResources(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range resourcesResult.Resources {
		fmt.Printf("  %s (%s)  - %s\n", r.Name, r.URI, r.Description)
	}
	for resourcesResult.NextCursor != "" {
		resourcesResult, err = client.ListResources(ctx, &mcp.ListParams{Cursor: resourcesResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		for _, r := range resourcesResult.Resources {
			fmt.Printf("  %s (%s)  - %s\n", r.Name, r.URI, r.Description)
		}
	}

	// Read a resource.
	resource, err := client.ReadResource(ctx, "test://hello")
	if err != nil {
		log.Fatal(err)
	}
	for _, c := range resource.Contents {
		fmt.Printf("\n%s:\n  %s\n", c.URI, c.Text)
	}
}
