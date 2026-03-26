package mcp

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Protocol version constants for MCP.
const (
	// ProtocolVersion20241105 is the legacy protocol version using SSE transport.
	ProtocolVersion20241105 = "2024-11-05"

	// ProtocolVersion20250326 is the current protocol version using Streamable HTTP.
	ProtocolVersion20250326 = "2025-03-26"

	// DefaultProtocolVersion is the recommended protocol version for new connections.
	DefaultProtocolVersion = ProtocolVersion20250326
)

// JSON-RPC error codes defined by the specification.
const (
	// ErrCodeParseError indicates invalid JSON was received.
	ErrCodeParseError = -32700

	// ErrCodeInvalidRequest indicates the JSON sent is not a valid request.
	ErrCodeInvalidRequest = -32600

	// ErrCodeMethodNotFound indicates the method does not exist.
	ErrCodeMethodNotFound = -32601

	// ErrCodeInvalidParams indicates invalid method parameters.
	ErrCodeInvalidParams = -32602

	// ErrCodeInternalError indicates an internal JSON-RPC error.
	ErrCodeInternalError = -32603

	// ErrCodeServerError indicates an MCP-specific server error.
	ErrCodeServerError = -32000
)

// InitializeParams contains the parameters sent during the initialize handshake.
type InitializeParams struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      ClientInfo         `json:"clientInfo"`
}

// InitializeResult contains the server's response to the initialize handshake.
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
	Instructions    string             `json:"instructions,omitempty"`
}

// ListParams contains optional pagination parameters for list methods.
type ListParams struct {
	Cursor string `json:"cursor,omitempty"`
}

// ClientInfo identifies the MCP client to the server.
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerInfo identifies the MCP server to the client.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ClientCapabilities describes the capabilities the client supports.
// Currently empty as capability negotiation is server-driven.
type ClientCapabilities struct{}

// ServerCapabilities describes the capabilities the server supports.
type ServerCapabilities struct {
	Tools     *ToolsCapability     `json:"tools,omitempty"`
	Prompts   *PromptsCapability   `json:"prompts,omitempty"`
	Resources *ResourcesCapability `json:"resources,omitempty"`
}

// ToolsCapability indicates the server supports tool operations.
type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// PromptsCapability indicates the server supports prompt operations.
type PromptsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// ResourcesCapability indicates the server supports resource operations.
type ResourcesCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
	Subscribe   bool `json:"subscribe,omitempty"`
}

// Tool represents an MCP tool definition returned by the server.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations *ToolAnnotations `json:"annotations,omitempty"`
}

// ToolAnnotations provides optional hints about tool behavior.
type ToolAnnotations struct {
	Title           string `json:"title,omitempty"`
	ReadOnlyHint    *bool  `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool  `json:"destructiveHint,omitempty"`
	IdempotentHint  *bool  `json:"idempotentHint,omitempty"`
	OpenWorldHint   *bool  `json:"openWorldHint,omitempty"`
}

// CallToolResult contains the result of calling an MCP tool.
type CallToolResult struct {
	Content           []ContentBlock  `json:"content"`
	IsError           bool            `json:"isError,omitempty"`
	StructuredContent json.RawMessage `json:"structuredContent,omitempty"`
}

// ContentBlock is a raw JSON block representing tool output content.
// Use ParseTextContent to extract text from a content block.
type ContentBlock = json.RawMessage

// TextContent represents text output from a tool.
type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ImageContent represents base64-encoded image output from a tool.
type ImageContent struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MIMEType string `json:"mimeType"`
}

// EmbeddedResource represents resource output embedded in a tool result.
type EmbeddedResource struct {
	Type     string          `json:"type"`
	Resource ResourceContent `json:"resource"`
}

// ParseTextContent attempts to parse a ContentBlock as TextContent.
// Returns the parsed content and true if the block is a text content block.
func ParseTextContent(block ContentBlock) (TextContent, bool) {
	var tc TextContent
	if err := json.Unmarshal(block, &tc); err != nil {
		return TextContent{}, false
	}
	if tc.Type != "text" {
		return TextContent{}, false
	}
	return tc, true
}

// Prompt represents an MCP prompt template returned by the server.
type Prompt struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Arguments   []PromptArgument `json:"arguments,omitempty"`
}

// PromptArgument describes a parameter for an MCP prompt.
type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

// GetPromptResult contains the server's response to a prompts/get request.
type GetPromptResult struct {
	Description string          `json:"description,omitempty"`
	Messages    []PromptMessage `json:"messages"`
}

// PromptMessage represents a message in a prompt template.
type PromptMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// Resource represents an MCP resource exposed by the server.
type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MIMEType    string `json:"mimeType,omitempty"`
}

// ReadResourceResult contains the server's response to a resources/read request.
type ReadResourceResult struct {
	Contents []ResourceContent `json:"contents"`
}

// ResourceContent represents a single resource's content returned by resources/read.
type ResourceContent struct {
	URI      string `json:"uri"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"`
	MIMEType string `json:"mimeType,omitempty"`
}

// ListToolsResult contains the response from tools/list including pagination.
type ListToolsResult struct {
	Tools      []Tool `json:"tools"`
	NextCursor string `json:"nextCursor,omitempty"`
}

// ListPromptsResult contains the response from prompts/list including pagination.
type ListPromptsResult struct {
	Prompts    []Prompt `json:"prompts"`
	NextCursor string   `json:"nextCursor,omitempty"`
}

// ListResourcesResult contains the response from resources/list including pagination.
type ListResourcesResult struct {
	Resources  []Resource `json:"resources"`
	NextCursor string     `json:"nextCursor,omitempty"`
}

// MCPError represents an error returned by an MCP server via JSON-RPC.
type MCPError struct {
	Code    int
	Message string
	Data    json.RawMessage
	cause   error // cached, built once at construction
}

// newMCPError creates an MCPError with a pre-built cause for Unwrap stability.
func newMCPError(code int, message string, data json.RawMessage) *MCPError {
	e := &MCPError{
		Code:    code,
		Message: message,
		Data:    data,
	}
	e.cause = e.buildCause()
	return e
}

// buildCause constructs the underlying cause error once.
func (e *MCPError) buildCause() error {
	if len(e.Data) > 0 {
		return fmt.Errorf("%s: %s", e.Message, string(e.Data))
	}
	return errors.New(e.Message)
}

// Error returns a human-readable representation of the MCP error.
func (e *MCPError) Error() string {
	return fmt.Sprintf("MCP error %d: %s", e.Code, e.Message)
}

// Unwrap returns the cached underlying error with additional data context if present.
func (e *MCPError) Unwrap() error {
	return e.cause
}
