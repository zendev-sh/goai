package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Client is an MCP protocol client that communicates with an MCP server
// over a pluggable transport. It handles the JSON-RPC 2.0 request/response
// lifecycle, protocol initialization, and capability negotiation.
type Client struct {
	name    string
	version string

	transport       Transport
	capabilities    ServerCapabilities
	serverInfo      ServerInfo
	instructions    string
	protocolVersion string
	initialized     atomic.Bool

	pendingMu sync.Mutex
	pending   map[string]chan JSONRPCMessage
	nextID    atomic.Int64

	requestTimeout time.Duration

	mu                 sync.Mutex
	onToolsChanged     func()
	onPromptsChanged   func()
	onResourcesChanged func()
	onLogMessage       func(level, logger, data string)
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithRequestTimeout sets the maximum duration to wait for a server response.
// The default is 60 seconds.
func WithRequestTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) { c.requestTimeout = timeout }
}

// WithTransport sets the transport used for communication.
func WithTransport(t Transport) ClientOption {
	return func(c *Client) { c.transport = t }
}

// NewClient creates a new MCP client with the given name and version.
// The transport must be provided via WithTransport.
func NewClient(name, version string, opts ...ClientOption) *Client {
	c := &Client{
		name:           name,
		version:        version,
		pending:        make(map[string]chan JSONRPCMessage),
		requestTimeout: 60 * time.Second,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Connect starts the transport, registers the message handler, and performs
// the MCP initialize handshake with the server.
func (c *Client) Connect(ctx context.Context) error {
	if c.transport == nil {
		return fmt.Errorf("mcp: no transport configured")
	}

	c.transport.OnMessage(c.handleMessage)

	if err := c.transport.Start(ctx); err != nil {
		return fmt.Errorf("mcp: transport start: %w", err)
	}

	return c.initialize(ctx)
}

// Close shuts down the transport and cleans up pending requests.
func (c *Client) Close() error {
	// Reject all pending requests
	c.pendingMu.Lock()
	for id, ch := range c.pending {
		close(ch)
		delete(c.pending, id)
	}
	c.pendingMu.Unlock()

	if c.transport != nil {
		return c.transport.Close()
	}
	return nil
}

// Ping sends a ping request to the server.
func (c *Client) Ping(ctx context.Context) error {
	if c.transport == nil {
		return fmt.Errorf("mcp: no transport configured")
	}
	_, err := c.sendRequest(ctx, "ping", struct{}{})
	return err
}

// Instructions returns the server's instructions from the initialize handshake.
func (c *Client) Instructions() string {
	return c.instructions
}

// ServerCapabilities returns the capabilities reported by the server during
// the initialize handshake.
func (c *Client) ServerCapabilities() ServerCapabilities {
	return c.capabilities
}

// ServerInfo returns the server's identity information.
func (c *Client) ServerInfo() ServerInfo {
	return c.serverInfo
}

// ProtocolVersion returns the protocol version negotiated with the server.
func (c *Client) ProtocolVersion() string {
	return c.protocolVersion
}

// OnToolsChanged registers a callback invoked when the server sends a
// notifications/tools/list_changed notification.
func (c *Client) OnToolsChanged(handler func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onToolsChanged = handler
}

// OnPromptsChanged registers a callback invoked when the server sends a
// notifications/prompts/list_changed notification.
func (c *Client) OnPromptsChanged(handler func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onPromptsChanged = handler
}

// OnResourcesChanged registers a callback invoked when the server sends a
// notifications/resources/list_changed notification.
func (c *Client) OnResourcesChanged(handler func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onResourcesChanged = handler
}

// OnLogMessage registers a callback invoked when the server sends a
// notifications/message notification. The callback receives the log level,
// logger name, and data string.
func (c *Client) OnLogMessage(handler func(level, logger, data string)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onLogMessage = handler
}

// ListTools requests the list of available tools from the server.
func (c *Client) ListTools(ctx context.Context, params *ListParams) (*ListToolsResult, error) {
	if err := c.assertCapability("tools"); err != nil {
		return nil, err
	}
	var p any = struct{}{}
	if params != nil {
		p = params
	}
	resp, err := c.sendRequest(ctx, "tools/list", p)
	if err != nil {
		return nil, err
	}
	var result ListToolsResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal tools/list result: %w", err)
	}
	return &result, nil
}

// CallTool invokes a named tool on the server with the given arguments.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	if err := c.assertCapability("tools"); err != nil {
		return nil, err
	}
	params := struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments,omitempty"`
	}{
		Name:      name,
		Arguments: args,
	}
	resp, err := c.sendRequest(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}
	var result CallToolResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal tools/call result: %w", err)
	}
	return &result, nil
}

// ListPrompts requests the list of available prompts from the server.
func (c *Client) ListPrompts(ctx context.Context, params *ListParams) (*ListPromptsResult, error) {
	if err := c.assertCapability("prompts"); err != nil {
		return nil, err
	}
	var p any = struct{}{}
	if params != nil {
		p = params
	}
	resp, err := c.sendRequest(ctx, "prompts/list", p)
	if err != nil {
		return nil, err
	}
	var result ListPromptsResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal prompts/list result: %w", err)
	}
	return &result, nil
}

// GetPrompt retrieves a specific prompt by name with the given arguments.
func (c *Client) GetPrompt(ctx context.Context, name string, args map[string]string) (*GetPromptResult, error) {
	if err := c.assertCapability("prompts"); err != nil {
		return nil, err
	}
	params := struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments,omitempty"`
	}{
		Name:      name,
		Arguments: args,
	}
	resp, err := c.sendRequest(ctx, "prompts/get", params)
	if err != nil {
		return nil, err
	}
	var result GetPromptResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal prompts/get result: %w", err)
	}
	return &result, nil
}

// ListResources requests the list of available resources from the server.
func (c *Client) ListResources(ctx context.Context, params *ListParams) (*ListResourcesResult, error) {
	if err := c.assertCapability("resources"); err != nil {
		return nil, err
	}
	var p any = struct{}{}
	if params != nil {
		p = params
	}
	resp, err := c.sendRequest(ctx, "resources/list", p)
	if err != nil {
		return nil, err
	}
	var result ListResourcesResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal resources/list result: %w", err)
	}
	return &result, nil
}

// ReadResource reads the content of a resource identified by URI.
func (c *Client) ReadResource(ctx context.Context, uri string) (*ReadResourceResult, error) {
	if err := c.assertCapability("resources"); err != nil {
		return nil, err
	}
	params := struct {
		URI string `json:"uri"`
	}{
		URI: uri,
	}
	resp, err := c.sendRequest(ctx, "resources/read", params)
	if err != nil {
		return nil, err
	}
	var result ReadResourceResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: unmarshal resources/read result: %w", err)
	}
	return &result, nil
}

// initialize performs the MCP handshake: sends initialize request, stores
// server capabilities, and sends the initialized notification.
func (c *Client) initialize(ctx context.Context) error {
	params := InitializeParams{
		ProtocolVersion: DefaultProtocolVersion,
		Capabilities:    ClientCapabilities{},
		ClientInfo: ClientInfo{
			Name:    c.name,
			Version: c.version,
		},
	}

	resp, err := c.sendRequest(ctx, "initialize", params)
	if err != nil {
		return fmt.Errorf("mcp: initialize: %w", err)
	}

	var result InitializeResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return fmt.Errorf("mcp: unmarshal initialize result: %w", err)
	}

	// Validate protocol version
	switch result.ProtocolVersion {
	case ProtocolVersion20250326, ProtocolVersion20241105:
		// supported
	default:
		return fmt.Errorf("mcp: unsupported protocol version %q from server", result.ProtocolVersion)
	}

	c.protocolVersion = result.ProtocolVersion
	c.capabilities = result.Capabilities
	c.serverInfo = result.ServerInfo
	c.instructions = result.Instructions

	// Send initialized notification (no ID = notification)
	notif := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}
	if err := c.transport.Send(ctx, notif); err != nil {
		return fmt.Errorf("mcp: send initialized notification: %w", err)
	}

	c.initialized.Store(true)
	return nil
}

// sendRequest sends a JSON-RPC request and waits for the matching response.
// It handles timeout, cleanup, and conversion of JSON-RPC errors to MCPError.
func (c *Client) sendRequest(ctx context.Context, method string, params any) (JSONRPCMessage, error) {
	id := c.generateID()

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return JSONRPCMessage{}, fmt.Errorf("mcp: marshal params: %w", err)
	}

	msg := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  method,
		Params:  paramsJSON,
		ID:      id,
	}

	ch := make(chan JSONRPCMessage, 1)
	c.pendingMu.Lock()
	c.pending[id] = ch
	c.pendingMu.Unlock()

	cleanup := func() {
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
	}

	ctx, cancel := context.WithTimeout(ctx, c.requestTimeout)
	defer cancel()

	if err := c.transport.Send(ctx, msg); err != nil {
		cleanup()
		return JSONRPCMessage{}, fmt.Errorf("mcp: send %s: %w", method, err)
	}

	select {
	case <-ctx.Done():
		cleanup()
		return JSONRPCMessage{}, fmt.Errorf("mcp: %s: %w", method, ctx.Err())
	case resp, ok := <-ch:
		cleanup()
		if !ok {
			return JSONRPCMessage{}, fmt.Errorf("mcp: connection closed")
		}
		if resp.RPCError != nil {
			return JSONRPCMessage{}, newMCPError(resp.RPCError.Code, resp.RPCError.Message, resp.RPCError.Data)
		}
		return resp, nil
	}
}

// generateID returns a unique string ID for JSON-RPC requests.
func (c *Client) generateID() string {
	return strconv.FormatInt(c.nextID.Add(1), 10)
}

// handleMessage routes incoming JSON-RPC messages to the appropriate handler.
// Responses (messages with an ID and no method) are routed to pending request
// channels. Notifications (messages with a method and no ID) trigger registered
// callbacks.
func (c *Client) handleMessage(msg JSONRPCMessage) {
	// Response: has result or error, and no method (may have ID or id:null for parse errors).
	if msg.Method == "" {
		if msg.ID != nil {
			// Normalize ID to string for map lookup
			idStr := normalizeID(msg.ID)

			c.pendingMu.Lock()
			ch, ok := c.pending[idStr]
			c.pendingMu.Unlock()

			if ok {
				ch <- msg
			}
		}
		// id:null with error = server parse error; we cannot route without an ID.
		return
	}

	// Notification: has method
	c.handleNotification(msg)
}

// handleNotification processes server-initiated notifications.
func (c *Client) handleNotification(msg JSONRPCMessage) {
	switch msg.Method {
	case "notifications/tools/list_changed":
		c.mu.Lock()
		fn := c.onToolsChanged
		c.mu.Unlock()
		if fn != nil {
			fn()
		}

	case "notifications/prompts/list_changed":
		c.mu.Lock()
		fn := c.onPromptsChanged
		c.mu.Unlock()
		if fn != nil {
			fn()
		}

	case "notifications/resources/list_changed":
		c.mu.Lock()
		fn := c.onResourcesChanged
		c.mu.Unlock()
		if fn != nil {
			fn()
		}

	case "notifications/message":
		c.mu.Lock()
		fn := c.onLogMessage
		c.mu.Unlock()
		if fn != nil {
			var params struct {
				Level  string          `json:"level"`
				Logger string          `json:"logger"`
				Data   json.RawMessage `json:"data"`
			}
			if err := json.Unmarshal(msg.Params, &params); err == nil {
				// Data can be any JSON value per MCP spec; convert to string.
				data := string(params.Data)
				// Unwrap JSON strings (remove surrounding quotes).
				var s string
				if json.Unmarshal(params.Data, &s) == nil {
					data = s
				}
				fn(params.Level, params.Logger, data)
			}
		}
	}
}

// assertCapability checks whether the server advertised the given capability
// during initialization. Returns an error if the capability is not supported.
func (c *Client) assertCapability(capability string) error {
	switch capability {
	case "tools":
		if c.capabilities.Tools == nil {
			return fmt.Errorf("mcp: server does not support %s", capability)
		}
	case "prompts":
		if c.capabilities.Prompts == nil {
			return fmt.Errorf("mcp: server does not support %s", capability)
		}
	case "resources":
		if c.capabilities.Resources == nil {
			return fmt.Errorf("mcp: server does not support %s", capability)
		}
	default:
		return fmt.Errorf("mcp: unknown capability %q", capability)
	}
	return nil
}

// normalizeID converts a JSON-RPC ID (which may be a string or float64 after
// JSON unmarshaling) to a consistent string representation for map lookup.
func normalizeID(id any) string {
	switch v := id.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatInt(int64(v), 10)
	case json.Number:
		return v.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}
