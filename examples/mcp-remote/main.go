//go:build ignore

// Example: MCP over Streamable HTTP transport.
//
// Starts the built-in testserver in HTTP mode, then connects to it via
// HTTPTransport to demonstrate session management, notifications, and
// remote tool calling.
//
// Usage:
//
//	go run ./examples/mcp-remote
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// --- Start testserver in HTTP mode ---
	fmt.Println("Starting testserver in HTTP mode...")
	cmd := exec.CommandContext(ctx, "go", "run", "./mcp/testserver", "--mode=http", "--addr=:0")
	cmd.Stderr = os.Stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	defer func() {
		cmd.Process.Kill() //nolint:errcheck
		cmd.Wait()         //nolint:errcheck
	}()

	// Read the LISTENING:<addr> line to discover the server address.
	scanner := bufio.NewScanner(stdout)
	var addr string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "LISTENING:") {
			addr = strings.TrimPrefix(line, "LISTENING:")
			break
		}
	}
	if addr == "" {
		log.Fatal("testserver did not report its address")
	}
	url := "http://" + addr + "/"
	fmt.Printf("Testserver listening at %s\n\n", url)

	// --- Connect via HTTP transport ---
	transport := mcp.NewHTTPTransport(url)
	client := mcp.NewClient("goai-mcp-remote", "1.0.0", mcp.WithTransport(transport))

	// Register notification handlers before connecting.
	client.OnToolsChanged(func() {
		fmt.Println("[notification] tools list changed")
	})
	client.OnLogMessage(func(level, logger, data string) {
		fmt.Printf("[log] %s/%s: %s\n", level, logger, data)
	})

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

	// --- Discover and call tools ---
	fmt.Println("\n=== Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
	}
	// Paginate.
	for toolsResult.NextCursor != "" {
		toolsResult, err = client.ListTools(ctx, &mcp.ListParams{Cursor: toolsResult.NextCursor})
		if err != nil {
			log.Fatal(err)
		}
		for _, tool := range toolsResult.Tools {
			fmt.Printf("  %s  - %s\n", tool.Name, tool.Description)
		}
	}

	// Call tools over HTTP.
	echoResult, err := client.CallTool(ctx, "echo", map[string]any{
		"message": "Hello over HTTP!",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\necho(\"Hello over HTTP!\") = %s\n", mcp.FormatContent(echoResult.Content, echoResult.IsError))

	addResult, err := client.CallTool(ctx, "add", map[string]any{
		"a": 100.0,
		"b": 200.0,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("add(100, 200) = %s\n", mcp.FormatContent(addResult.Content, addResult.IsError))

	// --- Ping ---
	if err := client.Ping(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nPing: OK")
}
