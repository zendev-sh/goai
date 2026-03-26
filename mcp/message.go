// Package mcp implements a Model Context Protocol (MCP) client for GoAI.
//
// MCP is an open protocol that standardizes how AI applications connect to
// external tools and data sources. This package provides a pure-Go,
// transport-agnostic client that works with any MCP server.
//
// # Transports
//
// Three transports are supported:
//
//   - [StdioTransport]: communicates with a local MCP server via stdin/stdout of a child process.
//   - [HTTPTransport]: Streamable HTTP (MCP 2025-03-26) for remote servers.
//   - [SSETransport]: Legacy SSE (MCP 2024-11-05) for older servers.
//
// # Quick Start
//
// Connect to an MCP server, list tools, and call one:
//
//	transport := mcp.NewStdioTransport("npx", []string{"-y", "@modelcontextprotocol/server-filesystem", "."})
//	client := mcp.NewClient("my-app", "1.0", mcp.WithTransport(transport))
//	if err := client.Connect(ctx); err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
//	result, _ := client.ListTools(ctx, nil)
//	for _, tool := range result.Tools {
//	    fmt.Println(tool.Name, " -", tool.Description)
//	}
//
// # GoAI Integration
//
// Use [ConvertTools] to convert MCP tools into [goai.Tool] values for use
// with [goai.GenerateText] and [goai.StreamText]:
//
//	tools := mcp.ConvertTools(client, result.Tools)
//	res, _ := goai.GenerateText(ctx, model,
//	    goai.WithTools(tools...),
//	    goai.WithPrompt("List files in the current directory"),
//	    goai.WithMaxSteps(5),
//	)
//
// # Error Handling
//
// Server errors are returned as [*MCPError], which can be inspected with
// [errors.As]:
//
//	var mcpErr *mcp.MCPError
//	if errors.As(err, &mcpErr) {
//	    fmt.Printf("MCP error %d: %s\n", mcpErr.Code, mcpErr.Message)
//	}
package mcp

import "encoding/json"

// JSONRPCMessage represents a JSON-RPC 2.0 message used by the MCP protocol.
// It serves as request, response, and notification depending on which fields
// are populated.
type JSONRPCMessage struct {
	JSONRPC  string          `json:"jsonrpc"`
	Method   string          `json:"method,omitempty"`
	Params   json.RawMessage `json:"params,omitempty"`
	Result   json.RawMessage `json:"result,omitempty"`
	RPCError *JSONRPCError   `json:"error,omitempty"`
	ID       any             `json:"id"`
}

// MarshalJSON implements custom JSON marshaling to correctly handle the ID field.
// When ID is nil (notifications), the "id" field is omitted entirely.
// When ID is non-nil (including 0 or ""), the "id" field is always included.
func (m JSONRPCMessage) MarshalJSON() ([]byte, error) {
	if m.ID == nil {
		// Notification: omit id field entirely.
		type noID struct {
			JSONRPC  string          `json:"jsonrpc"`
			Method   string          `json:"method,omitempty"`
			Params   json.RawMessage `json:"params,omitempty"`
			Result   json.RawMessage `json:"result,omitempty"`
			RPCError *JSONRPCError   `json:"error,omitempty"`
		}
		return json.Marshal(noID{
			JSONRPC:  m.JSONRPC,
			Method:   m.Method,
			Params:   m.Params,
			Result:   m.Result,
			RPCError: m.RPCError,
		})
	}
	// Request/Response: include id even if 0 or "".
	type withID struct {
		JSONRPC  string          `json:"jsonrpc"`
		Method   string          `json:"method,omitempty"`
		Params   json.RawMessage `json:"params,omitempty"`
		Result   json.RawMessage `json:"result,omitempty"`
		RPCError *JSONRPCError   `json:"error,omitempty"`
		ID       any             `json:"id"`
	}
	return json.Marshal(withID{
		JSONRPC:  m.JSONRPC,
		Method:   m.Method,
		Params:   m.Params,
		Result:   m.Result,
		RPCError: m.RPCError,
		ID:       m.ID,
	})
}

// JSONRPCError represents an error object in a JSON-RPC 2.0 response.
type JSONRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}
