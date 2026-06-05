//go:build ignore

// Example: OAuth 2.1 + PKCE authentication for a remote MCP server.
//
// Connects to an OAuth-protected Streamable HTTP MCP server using the native
// OAuth support in the mcp package: authorization-server discovery (RFC 9728 →
// RFC 8414), dynamic client registration (RFC 7591), PKCE (RFC 7636), and
// automatic token refresh + retry-on-401.
//
// The example runs a temporary loopback HTTP server to capture the OAuth
// redirect, opens the authorization URL in your browser, and once authorized
// lists the server's tools. If GEMINI_API_KEY is set, it also runs a one-shot
// GenerateText call that lets Gemini use the MCP tools.
//
// Zero-config OAuth MCP servers (work without pre-registering a client):
//
//	https://mcp.linear.app/mcp       (Linear)
//	https://mcp.sentry.dev/mcp       (Sentry)
//	https://mcp.notion.com/mcp       (Notion)
//
// Usage:
//
//	go run ./examples/mcp-oauth/main.go https://mcp.linear.app/mcp
//	GEMINI_API_KEY=... go run ./examples/mcp-oauth/main.go https://mcp.sentry.dev/mcp
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/mcp"
	"github.com/zendev-sh/goai/provider/google"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run ./examples/mcp-oauth/main.go <mcp-server-url>")
	}
	serverURL := os.Args[1]

	ctx := context.Background()

	// --- Start a loopback server to capture the OAuth redirect ---
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	defer listener.Close()
	redirectURI := fmt.Sprintf("http://127.0.0.1:%d/callback", listener.Addr().(*net.TCPAddr).Port)

	redirectCh := make(chan string, 1)
	go http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { //nolint:errcheck
		if r.URL.Path != "/callback" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprintln(w, "Authorization complete. You can close this tab and return to the terminal.")
		redirectCh <- "http://127.0.0.1" + r.URL.RequestURI()
	}))

	// --- Build the OAuth token source ---
	// NewOAuthTokenSource discovers the authorization server, dynamically
	// registers a public client, and runs the PKCE authorization-code flow via
	// the Authorize callback below.
	ts, err := mcp.NewOAuthTokenSource(ctx, mcp.OAuthConfig{
		ServerURL:   serverURL,
		ClientName:  "goai-mcp-oauth-example",
		RedirectURI: redirectURI,
		Authorize: func(ctx context.Context, authURL string) (string, error) {
			fmt.Printf("\nOpen this URL to authorize (it should open automatically):\n\n  %s\n\n", authURL)
			openBrowser(authURL)
			select {
			case redirect := <-redirectCh:
				return redirect, nil
			case <-time.After(3 * time.Minute):
				return "", fmt.Errorf("timed out waiting for authorization")
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	})
	if err != nil {
		log.Fatalf("oauth setup: %v", err)
	}

	// --- Connect over Streamable HTTP with OAuth-authenticated requests ---
	// NewOAuthHTTPClient injects the bearer token and retries once on HTTP 401
	// after refreshing the token.
	httpClient := mcp.NewOAuthHTTPClient(ts, nil)
	transport := mcp.NewHTTPTransport(serverURL, mcp.WithHTTPClient(httpClient))
	client := mcp.NewClient("goai-mcp-oauth", "1.0.0", mcp.WithTransport(transport))

	fmt.Println("Connecting (a browser window will open for authorization)...")
	if err := client.Connect(ctx); err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer client.Close()

	info := client.ServerInfo()
	fmt.Printf("\nConnected to %s v%s\n\n", info.Name, info.Version)

	// --- List tools ---
	fmt.Println("=== Available Tools ===")
	toolsResult, err := client.ListTools(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, tool := range toolsResult.Tools {
		fmt.Printf("  %-40s %s\n", tool.Name, truncate(tool.Description, 60))
	}
	fmt.Printf("\nTotal: %d tools\n", len(toolsResult.Tools))

	// --- Optional: let Gemini use the MCP tools ---
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("\n(Set GEMINI_API_KEY to run a model call using these tools.)")
		return
	}

	fmt.Println("\n=== Gemini using MCP tools ===")
	model := google.Chat("gemini-2.5-flash", google.WithAPIKey(apiKey))
	goaiTools := mcp.ConvertTools(client, toolsResult.Tools)

	res, err := goai.GenerateText(ctx, model,
		goai.WithTools(goaiTools...),
		goai.WithPrompt("Briefly, what can you help me with using the tools available?"),
		goai.WithMaxSteps(5),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}

// openBrowser tries to open url in the default browser; failures are ignored
// since the URL is also printed for manual opening.
func openBrowser(url string) {
	var cmd string
	var args []string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
	case "windows":
		cmd, args = "rundll32", []string{"url.dll,FileProtocolHandler"}
	default:
		cmd = "xdg-open"
	}
	_ = exec.Command(cmd, append(args, url)...).Start()
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
