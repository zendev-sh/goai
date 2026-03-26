package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockTransport implements Transport for testing. It records sent messages,
// allows injecting responses, and supports callback registration.
type mockTransport struct {
	mu        sync.Mutex
	sent      []JSONRPCMessage
	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)

	startErr error
	sendErr  error
	closeErr error

	// sendFunc, if set, overrides default send behavior.
	sendFunc func(ctx context.Context, msg JSONRPCMessage) error

	// autoRespond, if true, calls respondToInit automatically.
	autoRespond     bool
	serverCaps      ServerCapabilities
	serverInfo      ServerInfo
	serverInstructs string
}

func newMockTransport() *mockTransport {
	return &mockTransport{
		autoRespond: true,
		serverCaps: ServerCapabilities{
			Tools:     &ToolsCapability{ListChanged: true},
			Prompts:   &PromptsCapability{ListChanged: true},
			Resources: &ResourcesCapability{ListChanged: true},
		},
		serverInfo: ServerInfo{Name: "test-server", Version: "1.0"},
	}
}

func (m *mockTransport) Start(_ context.Context) error { return m.startErr }
func (m *mockTransport) Close() error                   { return m.closeErr }

func (m *mockTransport) OnMessage(fn func(JSONRPCMessage)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onMessage = fn
}

func (m *mockTransport) OnClose(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onClose = fn
}

func (m *mockTransport) OnError(fn func(error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onError = fn
}

func (m *mockTransport) Send(ctx context.Context, msg JSONRPCMessage) error {
	if m.sendFunc != nil {
		return m.sendFunc(ctx, msg)
	}
	if m.sendErr != nil {
		return m.sendErr
	}

	m.mu.Lock()
	m.sent = append(m.sent, msg)
	handler := m.onMessage
	autoResp := m.autoRespond
	m.mu.Unlock()

	if autoResp && handler != nil && msg.ID != nil {
		switch msg.Method {
		case "initialize":
			result := InitializeResult{
				ProtocolVersion: DefaultProtocolVersion,
				Capabilities:    m.serverCaps,
				ServerInfo:      m.serverInfo,
				Instructions:    m.serverInstructs,
			}
			resultJSON, _ := json.Marshal(result)
			handler(JSONRPCMessage{
				JSONRPC: "2.0",
				ID:      msg.ID,
				Result:  resultJSON,
			})
		default:
			// Auto-respond with empty result for other methods
			handler(JSONRPCMessage{
				JSONRPC: "2.0",
				ID:      msg.ID,
				Result:  json.RawMessage(`{}`),
			})
		}
	}
	return nil
}

// inject dispatches a message to the registered handler.
func (m *mockTransport) inject(msg JSONRPCMessage) {
	m.mu.Lock()
	fn := m.onMessage
	m.mu.Unlock()
	if fn != nil {
		fn(msg)
	}
}

// lastSent returns the most recently sent message.
func (m *mockTransport) lastSent() JSONRPCMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.sent) == 0 {
		return JSONRPCMessage{}
	}
	return m.sent[len(m.sent)-1]
}

// sentMessages returns a copy of all sent messages.
func (m *mockTransport) sentMessages() []JSONRPCMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]JSONRPCMessage, len(m.sent))
	copy(cp, m.sent)
	return cp
}

// --- Tests ---

func TestNewClient_Defaults(t *testing.T) {
	c := NewClient("test", "1.0")
	if c.name != "test" {
		t.Errorf("name = %q, want %q", c.name, "test")
	}
	if c.version != "1.0" {
		t.Errorf("version = %q, want %q", c.version, "1.0")
	}
	if c.requestTimeout != 60*time.Second {
		t.Errorf("requestTimeout = %v, want %v", c.requestTimeout, 60*time.Second)
	}
	if c.pending == nil {
		t.Error("pending map should be initialized")
	}
	if c.transport != nil {
		t.Error("transport should be nil by default")
	}
}

func TestNewClient_WithOptions(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0",
		WithRequestTimeout(5*time.Second),
		WithTransport(mt),
	)
	if c.requestTimeout != 5*time.Second {
		t.Errorf("requestTimeout = %v, want %v", c.requestTimeout, 5*time.Second)
	}
	if c.transport != mt {
		t.Error("transport not set by WithTransport")
	}
}

func TestConnect_NilTransport(t *testing.T) {
	c := NewClient("test", "1.0")
	err := c.Connect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "no transport") {
		t.Errorf("expected 'no transport' error, got %v", err)
	}
}

func TestConnect_InitializeHandshake(t *testing.T) {
	mt := newMockTransport()
	mt.serverInfo = ServerInfo{Name: "my-server", Version: "2.0"}
	mt.serverInstructs = "Be helpful"
	c := NewClient("test-client", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Verify init params were sent
	msgs := mt.sentMessages()
	if len(msgs) < 2 {
		t.Fatalf("expected at least 2 sent messages (init + notification), got %d", len(msgs))
	}

	initMsg := msgs[0]
	if initMsg.Method != "initialize" {
		t.Errorf("first message method = %q, want %q", initMsg.Method, "initialize")
	}

	var initParams InitializeParams
	if err := json.Unmarshal(initMsg.Params, &initParams); err != nil {
		t.Fatalf("unmarshal init params: %v", err)
	}
	if initParams.ProtocolVersion != DefaultProtocolVersion {
		t.Errorf("protocol version = %q, want %q", initParams.ProtocolVersion, DefaultProtocolVersion)
	}
	if initParams.ClientInfo.Name != "test-client" {
		t.Errorf("client name = %q, want %q", initParams.ClientInfo.Name, "test-client")
	}

	// Verify initialized notification was sent
	notifMsg := msgs[1]
	if notifMsg.Method != "notifications/initialized" {
		t.Errorf("second message method = %q, want %q", notifMsg.Method, "notifications/initialized")
	}
	if notifMsg.ID != nil {
		t.Error("notification should not have an ID")
	}

	// Verify server info stored
	if c.ServerInfo().Name != "my-server" {
		t.Errorf("server name = %q, want %q", c.ServerInfo().Name, "my-server")
	}
	if c.Instructions() != "Be helpful" {
		t.Errorf("instructions = %q, want %q", c.Instructions(), "Be helpful")
	}
	if !c.initialized.Load() {
		t.Error("client should be marked as initialized")
	}
}

func TestConnect_TransportStartError(t *testing.T) {
	mt := newMockTransport()
	mt.startErr = errors.New("start failed")
	c := NewClient("test", "1.0", WithTransport(mt))

	err := c.Connect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "transport start") {
		t.Errorf("expected transport start error, got %v", err)
	}
}

func TestClose_RejectsPendingRequests(t *testing.T) {
	mt := newMockTransport()
	mt.autoRespond = false
	c := NewClient("test", "1.0", WithTransport(mt), WithRequestTimeout(5*time.Second))

	// Manually set up to bypass initialize
	c.transport.OnMessage(c.handleMessage)
	c.capabilities = ServerCapabilities{Tools: &ToolsCapability{}}

	// Start a request that will block
	errCh := make(chan error, 1)
	go func() {
		_, err := c.ListTools(context.Background(), nil)
		errCh <- err
	}()

	// Wait for the request to be registered
	time.Sleep(50 * time.Millisecond)

	// Close should reject pending
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The pending request should get an error (closed channel)
	select {
	case err := <-errCh:
		if err == nil {
			t.Error("expected error from rejected request")
		}
	case <-time.After(2 * time.Second):
		t.Error("timed out waiting for pending request to complete")
	}
}

func TestListTools_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Override send to return tools
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/list" {
			result, _ := json.Marshal(map[string]any{
				"tools": []map[string]any{
					{
						"name":        "calculator",
						"description": "Does math",
						"inputSchema": map[string]any{"type": "object"},
					},
				},
			})
			mt.inject(JSONRPCMessage{
				JSONRPC: "2.0",
				ID:      msg.ID,
				Result:  result,
			})
		}
		return nil
	}

	result, err := c.ListTools(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(result.Tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1", len(result.Tools))
	}
	if result.Tools[0].Name != "calculator" {
		t.Errorf("tool name = %q, want %q", result.Tools[0].Name, "calculator")
	}
	if result.Tools[0].Description != "Does math" {
		t.Errorf("tool description = %q, want %q", result.Tools[0].Description, "Does math")
	}
}

func TestListTools_WithPaginationCursor(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var sentParams json.RawMessage
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/list" {
			sentParams = msg.Params
			result, _ := json.Marshal(map[string]any{"tools": []any{}})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	_, err := c.ListTools(context.Background(), &ListParams{Cursor: "abc123"})
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}

	var p ListParams
	if err := json.Unmarshal(sentParams, &p); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	if p.Cursor != "abc123" {
		t.Errorf("cursor = %q, want %q", p.Cursor, "abc123")
	}
}

func TestListTools_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{} // no tools capability
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	_, err := c.ListTools(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestCallTool_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/call" {
			// Verify params
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			json.Unmarshal(msg.Params, &p)
			if p.Name != "calc" {
				t.Errorf("tool name = %q, want %q", p.Name, "calc")
			}

			result, _ := json.Marshal(CallToolResult{
				Content: []ContentBlock{
					json.RawMessage(`{"type":"text","text":"42"}`),
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	result, err := c.CallTool(context.Background(), "calc", map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(result.Content))
	}
}

func TestListPrompts_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "prompts/list" {
			result, _ := json.Marshal(map[string]any{
				"prompts": []map[string]any{
					{"name": "greeting", "description": "Says hello"},
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	result, err := c.ListPrompts(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListPrompts: %v", err)
	}
	if len(result.Prompts) != 1 || result.Prompts[0].Name != "greeting" {
		t.Errorf("prompts = %+v, want [{Name:greeting}]", result.Prompts)
	}
}

func TestGetPrompt_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var sentParams json.RawMessage
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "prompts/get" {
			sentParams = msg.Params
			result, _ := json.Marshal(GetPromptResult{
				Description: "A greeting",
				Messages: []PromptMessage{
					{Role: "user", Content: json.RawMessage(`"Hello"`)},
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	result, err := c.GetPrompt(context.Background(), "greeting", map[string]string{"name": "Alice"})
	if err != nil {
		t.Fatalf("GetPrompt: %v", err)
	}
	if result.Description != "A greeting" {
		t.Errorf("description = %q, want %q", result.Description, "A greeting")
	}

	// Verify sent params
	var p struct {
		Name      string            `json:"name"`
		Arguments map[string]string `json:"arguments"`
	}
	json.Unmarshal(sentParams, &p)
	if p.Name != "greeting" || p.Arguments["name"] != "Alice" {
		t.Errorf("params = %+v, want name=greeting args={name:Alice}", p)
	}
}

func TestListResources_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "resources/list" {
			result, _ := json.Marshal(map[string]any{
				"resources": []map[string]any{
					{"uri": "file:///a.txt", "name": "a.txt"},
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	result, err := c.ListResources(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}
	if len(result.Resources) != 1 || result.Resources[0].URI != "file:///a.txt" {
		t.Errorf("resources = %+v", result.Resources)
	}
}

func TestReadResource_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var sentParams json.RawMessage
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "resources/read" {
			sentParams = msg.Params
			result, _ := json.Marshal(ReadResourceResult{
				Contents: []ResourceContent{
					{URI: "file:///a.txt", Text: "hello"},
				},
			})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	result, err := c.ReadResource(context.Background(), "file:///a.txt")
	if err != nil {
		t.Fatalf("ReadResource: %v", err)
	}
	if len(result.Contents) != 1 || result.Contents[0].Text != "hello" {
		t.Errorf("contents = %+v", result.Contents)
	}

	var p struct {
		URI string `json:"uri"`
	}
	json.Unmarshal(sentParams, &p)
	if p.URI != "file:///a.txt" {
		t.Errorf("uri = %q, want %q", p.URI, "file:///a.txt")
	}
}

func TestPing_Success(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "ping" {
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: json.RawMessage(`{}`)})
		}
		return nil
	}

	if err := c.Ping(context.Background()); err != nil {
		t.Fatalf("Ping: %v", err)
	}
}

func TestSendRequest_Timeout(t *testing.T) {
	mt := newMockTransport()
	mt.autoRespond = false
	c := NewClient("test", "1.0", WithTransport(mt), WithRequestTimeout(50*time.Millisecond))

	// Set up manually to bypass initialize
	c.transport.OnMessage(c.handleMessage)
	c.capabilities = ServerCapabilities{Tools: &ToolsCapability{}}

	// Send never responds, so it should timeout
	_, err := c.ListTools(context.Background(), nil)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected DeadlineExceeded, got %v", err)
	}
}

func TestSendRequest_JSONRPCError(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "tools/list" {
			mt.inject(JSONRPCMessage{
				JSONRPC: "2.0",
				ID:      msg.ID,
				RPCError: &JSONRPCError{
					Code:    ErrCodeMethodNotFound,
					Message: "method not found",
					Data:    json.RawMessage(`"extra info"`),
				},
			})
		}
		return nil
	}

	_, err := c.ListTools(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error")
	}

	var mcpErr *MCPError
	if !errors.As(err, &mcpErr) {
		t.Fatalf("expected MCPError, got %T: %v", err, err)
	}
	if mcpErr.Code != ErrCodeMethodNotFound {
		t.Errorf("code = %d, want %d", mcpErr.Code, ErrCodeMethodNotFound)
	}
	if mcpErr.Message != "method not found" {
		t.Errorf("message = %q, want %q", mcpErr.Message, "method not found")
	}
	if len(mcpErr.Data) == 0 {
		t.Error("expected non-empty data")
	}
}

func TestSendRequest_SendError(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "ping" {
			return errors.New("send failed")
		}
		return nil
	}

	err := c.Ping(context.Background())
	if err == nil || !strings.Contains(err.Error(), "send failed") {
		t.Errorf("expected send error, got %v", err)
	}
}

func TestHandleMessage_RoutesResponse(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	// Register a pending request manually
	ch := make(chan JSONRPCMessage, 1)
	c.pending["42"] = ch

	c.handleMessage(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      "42",
		Result:  json.RawMessage(`{"ok":true}`),
	})

	select {
	case msg := <-ch:
		if string(msg.Result) != `{"ok":true}` {
			t.Errorf("result = %s, want {\"ok\":true}", msg.Result)
		}
	case <-time.After(time.Second):
		t.Error("timed out waiting for response")
	}
}

func TestHandleMessage_IgnoresUnknownID(t *testing.T) {
	c := NewClient("test", "1.0")

	// Should not panic
	c.handleMessage(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      "unknown-id",
		Result:  json.RawMessage(`{}`),
	})
}

func TestHandleMessage_FloatID(t *testing.T) {
	c := NewClient("test", "1.0")
	ch := make(chan JSONRPCMessage, 1)
	c.pending["42"] = ch

	// Server returns float64 ID (common after JSON unmarshal)
	c.handleMessage(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      float64(42),
		Result:  json.RawMessage(`{}`),
	})

	select {
	case <-ch:
		// success
	case <-time.After(time.Second):
		t.Error("float64 ID not routed correctly")
	}
}

func TestHandleMessage_JSONNumberID(t *testing.T) {
	c := NewClient("test", "1.0")
	ch := make(chan JSONRPCMessage, 1)
	c.pending["42"] = ch

	c.handleMessage(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      json.Number("42"),
		Result:  json.RawMessage(`{}`),
	})

	select {
	case <-ch:
		// success
	case <-time.After(time.Second):
		t.Error("json.Number ID not routed correctly")
	}
}

func TestOnToolsChanged(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var called atomic.Bool
	c.OnToolsChanged(func() {
		called.Store(true)
	})

	mt.inject(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/tools/list_changed",
	})

	time.Sleep(50 * time.Millisecond)
	if !called.Load() {
		t.Error("OnToolsChanged callback not invoked")
	}
}

func TestOnPromptsChanged(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var called atomic.Bool
	c.OnPromptsChanged(func() {
		called.Store(true)
	})

	mt.inject(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/prompts/list_changed",
	})

	time.Sleep(50 * time.Millisecond)
	if !called.Load() {
		t.Error("OnPromptsChanged callback not invoked")
	}
}

func TestOnResourcesChanged(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var called atomic.Bool
	c.OnResourcesChanged(func() {
		called.Store(true)
	})

	mt.inject(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/resources/list_changed",
	})

	time.Sleep(50 * time.Millisecond)
	if !called.Load() {
		t.Error("OnResourcesChanged callback not invoked")
	}
}

func TestOnLogMessage(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var gotLevel, gotLogger, gotData string
	var called atomic.Bool
	c.OnLogMessage(func(level, logger, data string) {
		gotLevel = level
		gotLogger = logger
		gotData = data
		called.Store(true)
	})

	params, _ := json.Marshal(map[string]string{"level": "info", "data": "hello world"})
	mt.inject(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/message",
		Params:  params,
	})

	time.Sleep(50 * time.Millisecond)
	if !called.Load() {
		t.Fatal("OnLogMessage callback not invoked")
	}
	if gotLevel != "info" {
		t.Errorf("level = %q, want %q", gotLevel, "info")
	}
	if gotData != "hello world" {
		t.Errorf("data = %q, want %q", gotData, "hello world")
	}
	if gotLogger != "" {
		t.Errorf("logger = %q, want %q", gotLogger, "")
	}
}

func TestOnLogMessage_NoHandler(t *testing.T) {
	c := NewClient("test", "1.0")

	params, _ := json.Marshal(map[string]string{"level": "info", "data": "test"})
	// Should not panic with no handler registered
	c.handleNotification(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/message",
		Params:  params,
	})
}

func TestNormalizeID(t *testing.T) {
	tests := []struct {
		input any
		want  string
	}{
		{"hello", "hello"},
		{float64(42), "42"},
		{json.Number("99"), "99"},
		{42, "42"},
	}
	for _, tt := range tests {
		got := normalizeID(tt.input)
		if got != tt.want {
			t.Errorf("normalizeID(%v) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestServerCapabilities_Getter(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{
		Tools: &ToolsCapability{ListChanged: true},
	}
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	caps := c.ServerCapabilities()
	if caps.Tools == nil {
		t.Error("expected tools capability")
	}
	if caps.Prompts != nil {
		t.Error("expected nil prompts capability")
	}
}

func TestServerInfo_Getter(t *testing.T) {
	mt := newMockTransport()
	mt.serverInfo = ServerInfo{Name: "srv", Version: "3.0"}
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	info := c.ServerInfo()
	if info.Name != "srv" || info.Version != "3.0" {
		t.Errorf("server info = %+v", info)
	}
}

func TestInstructions_Getter(t *testing.T) {
	mt := newMockTransport()
	mt.serverInstructs = "Follow these rules"
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	if c.Instructions() != "Follow these rules" {
		t.Errorf("instructions = %q, want %q", c.Instructions(), "Follow these rules")
	}
}

func TestConcurrentSendRequest(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))

	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "ping" {
			// Respond asynchronously
			go func() {
				time.Sleep(5 * time.Millisecond)
				mt.inject(JSONRPCMessage{
					JSONRPC: "2.0",
					ID:      msg.ID,
					Result:  json.RawMessage(`{}`),
				})
			}()
		}
		return nil
	}

	const n = 20
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			errs[idx] = c.Ping(context.Background())
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("request %d failed: %v", i, err)
		}
	}
}

func TestAssertCapability_Tools(t *testing.T) {
	c := NewClient("test", "1.0")
	c.capabilities = ServerCapabilities{}

	if err := c.assertCapability("tools"); err == nil {
		t.Error("expected error for missing tools capability")
	}

	c.capabilities.Tools = &ToolsCapability{}
	if err := c.assertCapability("tools"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAssertCapability_Prompts(t *testing.T) {
	c := NewClient("test", "1.0")
	c.capabilities = ServerCapabilities{}

	if err := c.assertCapability("prompts"); err == nil {
		t.Error("expected error for missing prompts capability")
	}

	c.capabilities.Prompts = &PromptsCapability{}
	if err := c.assertCapability("prompts"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAssertCapability_Resources(t *testing.T) {
	c := NewClient("test", "1.0")
	c.capabilities = ServerCapabilities{}

	if err := c.assertCapability("resources"); err == nil {
		t.Error("expected error for missing resources capability")
	}

	c.capabilities.Resources = &ResourcesCapability{}
	if err := c.assertCapability("resources"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAssertCapability_Unknown(t *testing.T) {
	c := NewClient("test", "1.0")
	// Unknown capability should return an error
	if err := c.assertCapability("unknown"); err == nil {
		t.Error("expected error for unknown capability")
	}
}

func TestMCPError(t *testing.T) {
	err := newMCPError(ErrCodeInternalError, "something broke", json.RawMessage(`"details"`))
	if !strings.Contains(err.Error(), "MCP error") {
		t.Errorf("Error() = %q", err.Error())
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("%d", ErrCodeInternalError)) {
		t.Errorf("Error() should contain code: %q", err.Error())
	}

	unwrapped := err.Unwrap()
	if unwrapped == nil {
		t.Fatal("Unwrap returned nil")
	}
	if !strings.Contains(unwrapped.Error(), "details") {
		t.Errorf("unwrapped error = %q, want to contain 'details'", unwrapped.Error())
	}
}

func TestMCPError_NoData(t *testing.T) {
	err := newMCPError(ErrCodeServerError, "server error", nil)
	unwrapped := err.Unwrap()
	if unwrapped == nil {
		t.Fatal("Unwrap returned nil")
	}
	if unwrapped.Error() != "server error" {
		t.Errorf("unwrapped = %q, want %q", unwrapped.Error(), "server error")
	}
}

func TestClose_NilTransport(t *testing.T) {
	c := NewClient("test", "1.0")
	if err := c.Close(); err != nil {
		t.Errorf("Close with nil transport should not error: %v", err)
	}
}

func TestHandleNotification_UnknownMethod(t *testing.T) {
	c := NewClient("test", "1.0")
	// Should not panic on unknown notification
	c.handleNotification(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/unknown",
	})
}

func TestSendRequest_ClosedChannel(t *testing.T) {
	mt := newMockTransport()
	mt.autoRespond = false
	c := NewClient("test", "1.0", WithTransport(mt), WithRequestTimeout(2*time.Second))

	c.transport.OnMessage(c.handleMessage)
	c.capabilities = ServerCapabilities{Tools: &ToolsCapability{}}

	errCh := make(chan error, 1)
	go func() {
		_, err := c.ListTools(context.Background(), nil)
		errCh <- err
	}()

	// Wait for request to be pending, then close the channel
	time.Sleep(50 * time.Millisecond)
	c.pendingMu.Lock()
	for _, ch := range c.pending {
		close(ch)
	}
	c.pendingMu.Unlock()

	select {
	case err := <-errCh:
		if err == nil {
			t.Error("expected error from closed channel")
		}
	case <-time.After(3 * time.Second):
		t.Error("timed out")
	}
}

func TestCallTool_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{}
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err := c.CallTool(context.Background(), "x", nil)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestListPrompts_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{}
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err := c.ListPrompts(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestGetPrompt_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{}
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err := c.GetPrompt(context.Background(), "x", nil)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestListResources_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{}
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err := c.ListResources(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestReadResource_NoCapability(t *testing.T) {
	mt := newMockTransport()
	mt.serverCaps = ServerCapabilities{}
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err := c.ReadResource(context.Background(), "file:///x")
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Errorf("expected capability error, got %v", err)
	}
}

func TestListPrompts_WithCursor(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var sentParams json.RawMessage
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "prompts/list" {
			sentParams = msg.Params
			result, _ := json.Marshal(map[string]any{"prompts": []any{}})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	_, err := c.ListPrompts(context.Background(), &ListParams{Cursor: "cur1"})
	if err != nil {
		t.Fatalf("ListPrompts: %v", err)
	}

	var p ListParams
	json.Unmarshal(sentParams, &p)
	if p.Cursor != "cur1" {
		t.Errorf("cursor = %q, want %q", p.Cursor, "cur1")
	}
}

func TestListResources_WithCursor(t *testing.T) {
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var sentParams json.RawMessage
	mt.sendFunc = func(_ context.Context, msg JSONRPCMessage) error {
		if msg.Method == "resources/list" {
			sentParams = msg.Params
			result, _ := json.Marshal(map[string]any{"resources": []any{}})
			mt.inject(JSONRPCMessage{JSONRPC: "2.0", ID: msg.ID, Result: result})
		}
		return nil
	}

	_, err := c.ListResources(context.Background(), &ListParams{Cursor: "cur2"})
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}

	var p ListParams
	json.Unmarshal(sentParams, &p)
	if p.Cursor != "cur2" {
		t.Errorf("cursor = %q, want %q", p.Cursor, "cur2")
	}
}

func TestPing_NilTransport(t *testing.T) {
	c := NewClient("test", "1.0")
	err := c.Ping(context.Background())
	if err == nil {
		t.Fatal("expected error for nil transport")
	}
}

func TestHandleMessage_NilIDErrorResponse(t *testing.T) {
	c := NewClient("test", "1.0")

	// Unmarshal a real JSON-RPC parse error response with "id": null
	raw := `{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}`
	var msg JSONRPCMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// ID should be nil after unmarshal of null
	if msg.ID != nil {
		t.Fatalf("expected nil ID, got %v", msg.ID)
	}
	// Should not panic or block
	c.handleMessage(msg)
}

func TestOnLogMessage_NonStringData(t *testing.T) {
	// MCP spec allows data to be any JSON value, not just string.
	mt := newMockTransport()
	c := NewClient("test", "1.0", WithTransport(mt))
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var gotLevel, gotLogger, gotData string
	var called atomic.Bool
	c.OnLogMessage(func(level, logger, data string) {
		gotLevel = level
		gotLogger = logger
		gotData = data
		called.Store(true)
	})

	// Send a log notification with object data
	mt.inject(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/message",
		Params:  json.RawMessage(`{"level":"error","data":{"key":"value"}}`),
	})

	time.Sleep(50 * time.Millisecond)
	if !called.Load() {
		t.Fatal("OnLogMessage callback not invoked for object data")
	}
	if gotLevel != "error" {
		t.Errorf("level = %q, want %q", gotLevel, "error")
	}
	// Non-string data should be rendered as JSON
	if gotData != `{"key":"value"}` {
		t.Errorf("data = %q, want %q", gotData, `{"key":"value"}`)
	}
	if gotLogger != "" {
		t.Errorf("logger = %q, want %q", gotLogger, "")
	}
}

func TestMarshalJSON_NotificationOmitsID(t *testing.T) {
	msg := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(data), `"id"`) {
		t.Errorf("notification should not contain id field, got: %s", data)
	}
}

func TestMarshalJSON_RequestPreservesZeroID(t *testing.T) {
	msg := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "test",
		ID:      float64(0),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"id":0`) {
		t.Errorf("request with id=0 should contain id field, got: %s", data)
	}
}

func TestMarshalJSON_RequestPreservesEmptyStringID(t *testing.T) {
	msg := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "test",
		ID:      "",
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"id":""`) {
		t.Errorf("request with id=\"\" should contain id field, got: %s", data)
	}
}

func TestInitialize_UnsupportedProtocolVersion(t *testing.T) {
	mt := newMockTransport()
	mt.sendFunc = func(ctx context.Context, msg JSONRPCMessage) error {
		mt.mu.Lock()
		onMsg := mt.onMessage
		mt.mu.Unlock()
		if msg.Method == "initialize" && onMsg != nil {
			resp := JSONRPCMessage{
				JSONRPC: "2.0",
				ID:      msg.ID,
				Result:  json.RawMessage(`{"protocolVersion":"1999-01-01","capabilities":{},"serverInfo":{"name":"test","version":"1.0"}}`),
			}
			go onMsg(resp)
		}
		return nil
	}
	c := NewClient("test", "1.0", WithTransport(mt))
	err := c.Connect(context.Background())
	if err == nil {
		t.Fatal("expected error for unsupported protocol version")
	}
	if !strings.Contains(err.Error(), "unsupported protocol version") {
		t.Errorf("error = %q, want to contain 'unsupported protocol version'", err.Error())
	}
}
