---
title: MCP Client
description: "Connect to MCP servers from Go. Use stdio, HTTP, or SSE transports to discover and call remote tools, then pass them to GenerateText or StreamText."
---

# MCP Client

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard for connecting AI applications to external tools and data sources. GoAI's `mcp` package provides a client that speaks MCP over pluggable transports, with a bridge to convert MCP tools into GoAI tools for use with `GenerateText` and `StreamText`.

## Quick Start

Connect to a local MCP server, list its tools, and call one directly:

```go
import "github.com/zendev-sh/goai/mcp"

// Create a client and connect via stdio transport.
client := mcp.NewClient("my-app", "1.0",
    mcp.WithTransport(mcp.NewStdioTransport("npx", []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"})),
)
if err := client.Connect(ctx); err != nil {
    log.Fatal(err)
}
defer client.Close()

// List available tools.
result, err := client.ListTools(ctx, nil)
if err != nil {
    log.Fatal(err)
}
for _, tool := range result.Tools {
    fmt.Printf("%s - %s\n", tool.Name, tool.Description)
}

// Call a tool directly.
callResult, err := client.CallTool(ctx, "read_file", map[string]any{"path": "/tmp/hello.txt"})
if err != nil {
    log.Fatal(err)
}
fmt.Println(mcp.FormatContent(callResult.Content, callResult.IsError))
```

## Transports

The client communicates with MCP servers through a `Transport` interface. Three built-in transports cover the common deployment patterns.

| Transport | Constructor                             | Use Case                                         |
| --------- | --------------------------------------- | ------------------------------------------------ |
| **Stdio** | `NewStdioTransport(cmd, args, opts...)` | Local servers run as child processes             |
| **HTTP**  | `NewHTTPTransport(url, opts...)`        | Remote servers (Streamable HTTP, MCP 2025-03-26) |
| **SSE**   | `NewSSETransport(url, opts...)`         | Remote servers (legacy SSE, MCP 2024-11-05)      |

### Stdio

Spawns a local process and communicates via stdin/stdout. Best for CLI tools and local development.

```go
transport := mcp.NewStdioTransport("npx", []string{"-y", "some-mcp-server"},
    mcp.WithStdioEnv(map[string]string{"API_KEY": os.Getenv("API_KEY")}),
    mcp.WithStdioDir("/working/dir"),
    mcp.WithStdioStderr(os.Stderr),
)
```

### HTTP (Streamable HTTP)

Uses HTTP POST for requests and GET SSE for server-initiated messages. This is the current recommended transport for remote servers.

```go
transport := mcp.NewHTTPTransport("https://mcp.example.com/mcp",
    mcp.WithHTTPHeaders(map[string]string{"Authorization": "Bearer " + token}),
    mcp.WithHTTPClient(customHTTPClient),
)
```

### SSE (Legacy)

Connects to an SSE endpoint for server messages and discovers a POST endpoint from the stream. Use this for servers that only support the 2024-11-05 protocol version.

```go
transport := mcp.NewSSETransport("https://mcp.example.com/sse",
    mcp.WithSSEHeaders(map[string]string{"Authorization": "Bearer " + token}),
    mcp.WithSSEHTTPClient(customHTTPClient),
)
```

## Using MCP Tools with GoAI

`mcp.ConvertTools` bridges MCP tools into `goai.Tool` values. Each converted tool's `Execute` function calls back to the MCP server via the client.

```go
import (
    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/mcp"
    "github.com/zendev-sh/goai/provider/openai"
)

// List tools from the MCP server.
result, _ := client.ListTools(ctx, nil)

// Convert to GoAI tools and pass to GenerateText.
tools := mcp.ConvertTools(client, result.Tools)

resp, err := goai.GenerateText(ctx, openai.Chat("gpt-4o"),
    goai.WithPrompt("List the files in /tmp"),
    goai.WithTools(tools...),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}
fmt.Println(resp.Text)
```

This works identically with `StreamText`.

## Pagination

List methods return a `NextCursor` field for paginated results. Pass it back via `ListParams`:

```go
var allTools []mcp.Tool
var cursor string
for {
    params := &mcp.ListParams{Cursor: cursor}
    if cursor == "" {
        params = nil
    }
    result, err := client.ListTools(ctx, params)
    if err != nil {
        log.Fatal(err)
    }
    allTools = append(allTools, result.Tools...)
    if result.NextCursor == "" {
        break
    }
    cursor = result.NextCursor
}
```

## Notification Handlers

MCP servers can push notifications when their capabilities change. Register handlers before calling `Connect`:

```go
client.OnToolsChanged(func() {
    fmt.Println("Server tools changed, re-fetching...")
    result, _ := client.ListTools(ctx, nil)
    // Update your tool set.
})

client.OnPromptsChanged(func() {
    fmt.Println("Prompts changed")
})

client.OnResourcesChanged(func() {
    fmt.Println("Resources changed")
})

client.OnLogMessage(func(level, logger, data string) {
    fmt.Printf("[%s] %s: %s\n", level, logger, data)
})
```

## Error Handling

MCP server errors are returned as `*mcp.MCPError`, which wraps JSON-RPC error codes:

```go
result, err := client.CallTool(ctx, "nonexistent", nil)
if err != nil {
    var mcpErr *mcp.MCPError
    if errors.As(err, &mcpErr) {
        fmt.Printf("MCP error %d: %s\n", mcpErr.Code, mcpErr.Message)
    }
}
```

Standard error codes:

| Code     | Constant                | Meaning                      |
| -------- | ----------------------- | ---------------------------- |
| `-32700` | `ErrCodeParseError`     | Invalid JSON received        |
| `-32600` | `ErrCodeInvalidRequest` | Not a valid JSON-RPC request |
| `-32601` | `ErrCodeMethodNotFound` | Method does not exist        |
| `-32602` | `ErrCodeInvalidParams`  | Invalid method parameters    |
| `-32603` | `ErrCodeInternalError`  | Internal server error        |
| `-32000` | `ErrCodeServerError`    | MCP-specific server error    |

## Examples

| Example                                                                                 | Description                                  |
| --------------------------------------------------------------------------------------- | -------------------------------------------- |
| [`mcp-local`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-local)           | Connect to a local stdio MCP server          |
| [`mcp-remote`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-remote)         | Connect to a remote HTTP MCP server          |
| [`mcp-sse`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-sse)               | Connect via legacy SSE transport             |
| [`mcp-tools`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-tools)           | Convert MCP tools for use with GenerateText  |
| [`mcp-filesystem`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-filesystem) | File operations via MCP filesystem server    |
| [`mcp-github`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-github)         | GitHub operations via MCP                    |
| [`mcp-playwright`](https://github.com/zendev-sh/goai/tree/main/examples/mcp-playwright) | Browser automation via Playwright MCP server |

> **Note:** Some MCP examples are marked with `//go:build ignore` and must be run by targeting the file directly (e.g. `go run ./examples/mcp-local/main.go`).

> **Note:** The `mcp-local`, `mcp-remote`, and `mcp-tools` examples use a local MCP test server command in each example (`go run ./mcp/testserver ...`). Ensure your local setup includes a compatible test server implementation.

> **Requirements:**
>
> - `mcp-github` requires `GITHUB_TOKEN` and Node.js (`npx`) before running.
> - `mcp-playwright` requires Node.js (`npx`) and may download browser binaries on first run.
