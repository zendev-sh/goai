//go:build ignore

// Example: anonymous web search and page extraction with Parallel Search MCP.
// No API key or language model is required.
//
// Usage (from the repository root):
//
//	go run ./examples/mcp-parallel/main.go -query "Go 1.25 release highlights"
//	go run ./examples/mcp-parallel/main.go -url https://go.dev/doc/go1.25
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/zendev-sh/goai/mcp"
)

const (
	parallelURL = "https://search.parallel.ai/mcp"
	userAgent   = "goai-mcp-parallel/1.0 (+https://github.com/zendev-sh/goai)"
)

func main() {
	query := flag.String("query", "Go 1.25 release highlights", "search objective and keyword query")
	url := flag.String("url", "", "fetch this HTTP(S) URL instead of searching")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if err := run(ctx, parallelURL, &http.Client{Timeout: 60 * time.Second}, *query, *url, os.Stdout); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run(ctx context.Context, endpoint string, httpClient *http.Client, query, url string, out io.Writer) error {
	transport := mcp.NewHTTPTransport(endpoint,
		mcp.WithHTTPClient(httpClient),
		mcp.WithHTTPHeaders(map[string]string{"User-Agent": userAgent}),
	)
	client := mcp.NewClient("goai-mcp-parallel", "1.0.0", mcp.WithTransport(transport))
	defer func() { _ = client.Close() }()
	if err := client.Connect(ctx); err != nil {
		return err
	}

	// Discover the tools and their input schemas before calling one.
	var cursor string
	for {
		tools, err := client.ListTools(ctx, &mcp.ListParams{Cursor: cursor})
		if err != nil {
			return err
		}
		for _, tool := range tools.Tools {
			if _, err := fmt.Fprintf(out, "Tool: %s\n", tool.Name); err != nil {
				return err
			}
		}
		cursor = tools.NextCursor
		if cursor == "" {
			break
		}
	}

	name := "web_search"
	args := map[string]any{"objective": query, "search_queries": []string{query}}
	if url != "" {
		name = "web_fetch"
		args = map[string]any{"urls": []string{url}}
	}
	result, err := client.CallTool(ctx, name, args)
	if err != nil {
		return err
	}
	if result.IsError {
		return fmt.Errorf("%s: %s", name, mcp.FormatContent(result.Content, false))
	}
	_, err = fmt.Fprintln(out, mcp.FormatContent(result.Content, false))
	return err
}
