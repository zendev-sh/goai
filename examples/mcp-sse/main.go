//go:build ignore

// Example: MCP over legacy SSE transport (protocol version 2024-11-05).
//
// Starts the built-in testserver in SSE mode, then connects via SSETransport.
// Demonstrates endpoint discovery, tool listing, and tool calling over the
// legacy SSE protocol.
//
// Usage:
//
//	go run ./examples/mcp-sse/main.go
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

	// --- Start testserver in SSE mode ---
	fmt.Println("Starting testserver in SSE mode...")
	cmd := exec.CommandContext(ctx, "go", "run", "./mcp/testserver", "--mode=sse", "--addr=:0")
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
	sseURL := "http://" + addr + "/sse"
	fmt.Printf("Testserver SSE endpoint: %s\n\n", sseURL)

	// --- Connect via legacy SSE transport ---
	// SSETransport connects to the SSE stream, waits for the server to send
	// an "endpoint" event with the POST URL, then uses that for all requests.
	transport := mcp.NewSSETransport(sseURL)

	client := mcp.NewClient("goai-mcp-sse", "1.0.0", mcp.WithTransport(transport))
	if err := client.Connect(ctx); err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("Connected to %s v%s\n", info.Name, info.Version)

	// --- List and call tools ---
	fmt.Println("\n=== Tools ===")
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

	echoResult, err := client.CallTool(ctx, "echo", map[string]any{
		"message": "Hello over SSE!",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\necho(\"Hello over SSE!\") = %s\n", mcp.FormatContent(echoResult.Content, echoResult.IsError))

	if err := client.Ping(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Ping: OK")
}
