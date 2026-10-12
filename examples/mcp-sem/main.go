//go:build ignore

// Example: inspect a local Git repository through Sem over stdio MCP.
// No model or API key is used. Requires npx; the first run downloads Sem.
//
// go run ./examples/mcp-sem/main.go -repo /absolute/path/to/repo -query parse_config
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/zendev-sh/goai/mcp"
)

func main() {
	if err := command(); err != nil {
		log.Fatal(err)
	}
}

func command() error {
	repo := flag.String("repo", "", "trusted Git checkout (required)")
	query := flag.String("query", "", "entity name to find (required)")
	flag.Parse()
	transport, err := semTransport(*repo, *query)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	return run(ctx, transport, *query, os.Stdout)
}

func semTransport(repo, query string) (*mcp.StdioTransport, error) {
	if strings.TrimSpace(repo) == "" || strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("-repo and -query are required")
	}
	if strings.ContainsAny(repo, "\n\r\x00") {
		return nil, fmt.Errorf("repository path contains unsupported control characters")
	}
	abs, err := filepath.Abs(repo)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("repository path must be a directory")
	}
	return mcp.NewStdioTransport("npx", []string{"-y", "@ataraxy-labs/sem@0.27.0", "mcp"},
		mcp.WithStdioDir(abs),
		mcp.WithStdioEnv(map[string]string{
			"SEM_REPO": abs, "SEM_NO_NETWORK": "1", "SEM_NO_TELEMETRY": "1",
		}),
		mcp.WithStdioStderr(os.Stderr)), nil
}

func run(ctx context.Context, transport mcp.Transport, query string, out io.Writer) error {
	client := mcp.NewClient("goai-mcp-sem", "1.0.0", mcp.WithTransport(transport))
	defer func() { _ = client.Close() }()
	if err := client.Connect(ctx); err != nil {
		return err
	}
	// Invoke only navigation, even though the server exposes other capabilities.
	result, err := client.CallTool(ctx, "sem_find", map[string]any{
		"query": query, "mode": "context", "token_budget": 1200,
	})
	if err != nil {
		return err
	}
	if result.IsError {
		return fmt.Errorf("sem_find: %s", mcp.FormatContent(result.Content, false))
	}
	_, err = fmt.Fprintln(out, mcp.FormatContent(result.Content, false))
	return err
}
