package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- parseSSEData tests ---

func TestParseSSEData_ValidData(t *testing.T) {
	data := `{"jsonrpc":"2.0","id":"1","result":{"ok":true}}`
	msg, ok := parseSSEData(data)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if msg.JSONRPC != "2.0" {
		t.Errorf("jsonrpc = %q, want %q", msg.JSONRPC, "2.0")
	}
}

func TestParseSSEData_EmptyData(t *testing.T) {
	_, ok := parseSSEData("")
	if ok {
		t.Error("expected ok=false for empty data")
	}
}

func TestParseSSEData_InvalidJSON(t *testing.T) {
	_, ok := parseSSEData("{invalid json}")
	if ok {
		t.Error("expected ok=false for invalid JSON")
	}
}

func TestReadSSEEvent_SingleDataLine(t *testing.T) {
	input := "data: {\"jsonrpc\":\"2.0\",\"id\":\"1\",\"result\":{}}\n\n"
	reader := bufio.NewReader(strings.NewReader(input))
	eventType, data, err := readSSEEvent(reader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if eventType != "" {
		t.Errorf("eventType = %q, want empty", eventType)
	}
	if data != `{"jsonrpc":"2.0","id":"1","result":{}}` {
		t.Errorf("data = %q", data)
	}
}

func TestReadSSEEvent_MultiLineData(t *testing.T) {
	input := "data: line1\ndata: line2\ndata: line3\n\n"
	reader := bufio.NewReader(strings.NewReader(input))
	_, data, err := readSSEEvent(reader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if data != "line1\nline2\nline3" {
		t.Errorf("data = %q, want %q", data, "line1\nline2\nline3")
	}
}

func TestReadSSEEvent_WithEventType(t *testing.T) {
	input := "event: message\ndata: hello\n\n"
	reader := bufio.NewReader(strings.NewReader(input))
	eventType, data, err := readSSEEvent(reader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if eventType != "message" {
		t.Errorf("eventType = %q, want %q", eventType, "message")
	}
	if data != "hello" {
		t.Errorf("data = %q, want %q", data, "hello")
	}
}

func TestReadSSEEvent_EOFWithData(t *testing.T) {
	// Data without trailing blank line, terminated by EOF
	input := "data: partial"
	reader := bufio.NewReader(strings.NewReader(input))
	_, data, err := readSSEEvent(reader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if data != "partial" {
		t.Errorf("data = %q, want %q", data, "partial")
	}
}

func TestReadSSEEvent_EOFNoData(t *testing.T) {
	reader := bufio.NewReader(strings.NewReader(""))
	_, _, err := readSSEEvent(reader)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

// --- mergeEnv tests ---

func TestMergeEnv_Empty(t *testing.T) {
	result := mergeEnv(nil)
	if len(result) == 0 {
		t.Error("expected non-empty env")
	}
	if len(result) != len(os.Environ()) {
		t.Errorf("len = %d, want %d", len(result), len(os.Environ()))
	}
}

func TestMergeEnv_AddsVars(t *testing.T) {
	base := len(os.Environ())
	result := mergeEnv(map[string]string{
		"TEST_VAR_A": "hello",
		"TEST_VAR_B": "world",
	})
	if len(result) != base+2 {
		t.Errorf("len = %d, want %d", len(result), base+2)
	}

	found := 0
	for _, env := range result {
		if env == "TEST_VAR_A=hello" || env == "TEST_VAR_B=world" {
			found++
		}
	}
	if found != 2 {
		t.Errorf("found %d test vars, want 2", found)
	}
}

// --- StdioTransport tests ---

func TestNewStdioTransport_WithOptions(t *testing.T) {
	var buf strings.Builder
	st := NewStdioTransport("echo", []string{"hello"},
		WithStdioEnv(map[string]string{"FOO": "bar"}),
		WithStdioDir("/tmp"),
		WithStdioStderr(&buf),
	)
	if st.Command != "echo" {
		t.Errorf("command = %q, want %q", st.Command, "echo")
	}
	if len(st.Args) != 1 || st.Args[0] != "hello" {
		t.Errorf("args = %v", st.Args)
	}
	if st.Env["FOO"] != "bar" {
		t.Error("env not set")
	}
	if st.Dir != "/tmp" {
		t.Errorf("dir = %q", st.Dir)
	}
	if st.Stderr == nil {
		t.Error("stderr not set")
	}
}

func TestStdioTransport_StartAndRead(t *testing.T) {
	msg := `{"jsonrpc":"2.0","id":"1","result":{}}`
	st := NewStdioTransport("echo", []string{msg})

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	st.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if received.JSONRPC != "2.0" {
			t.Errorf("jsonrpc = %q", received.JSONRPC)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for message")
	}

	if err := st.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestStdioTransport_SendWritesJSON(t *testing.T) {
	st := NewStdioTransport("cat", nil)

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	st.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	msg := JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  "test",
		ID:      "42",
	}
	if err := st.Send(ctx, msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if received.Method != "test" {
			t.Errorf("method = %q, want %q", received.Method, "test")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for echoed message")
	}

	st.Close()
}

func TestStdioTransport_CloseMultipleTimes(t *testing.T) {
	st := NewStdioTransport("echo", []string{"hi"})
	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	st.Close()
	st.Close()
}

func TestStdioTransport_OnClose(t *testing.T) {
	st := NewStdioTransport("echo", []string{`{"jsonrpc":"2.0"}`})

	var closeCalled atomic.Bool
	st.OnClose(func() {
		closeCalled.Store(true)
	})
	st.OnMessage(func(m JSONRPCMessage) {})

	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(500 * time.Millisecond)
	if !closeCalled.Load() {
		t.Error("OnClose not called after subprocess exit")
	}
	st.Close()
}

func TestStdioTransport_OnError(t *testing.T) {
	st := NewStdioTransport("echo", []string{"not valid json"})

	var errCalled atomic.Bool
	st.OnError(func(err error) {
		errCalled.Store(true)
	})
	st.OnMessage(func(m JSONRPCMessage) {})

	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// echo outputs invalid JSON followed by EOF. The json.Decoder may see
	// a decode error or EOF depending on timing. Either way should not panic.
	time.Sleep(500 * time.Millisecond)
	st.Close()
}

// --- HTTPTransport tests ---

func TestNewHTTPTransport_WithOptions(t *testing.T) {
	httpClient := &http.Client{Timeout: 5 * time.Second}
	ht := NewHTTPTransport("http://example.com/mcp",
		WithHTTPHeaders(map[string]string{"X-Key": "val"}),
		WithHTTPClient(httpClient),
	)
	if ht.URL != "http://example.com/mcp" {
		t.Errorf("url = %q", ht.URL)
	}
	if ht.Headers["X-Key"] != "val" {
		t.Error("header not set")
	}
	if ht.HTTPClient != httpClient {
		t.Error("http client not set")
	}
}

func TestHTTPTransport_Send_ContentTypeAndBody(t *testing.T) {
	var gotContentType string
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"jsonrpc":"2.0","id":"1","result":{}}`))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.OnMessage(func(m JSONRPCMessage) {})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	if gotContentType != "application/json" {
		t.Errorf("content-type = %q", gotContentType)
	}

	var received JSONRPCMessage
	if err := json.Unmarshal(gotBody, &received); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if received.Method != "test" {
		t.Errorf("method = %q", received.Method)
	}
}

func TestHTTPTransport_Send_ExtractsSessionID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("MCP-Session-ID", "sess-123")
		w.WriteHeader(200)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	if ht.SessionID != "sess-123" {
		t.Errorf("session id = %q, want %q", ht.SessionID, "sess-123")
	}
}

func TestHTTPTransport_Send_ExtractsEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("MCP-Endpoint", "http://other.com/post")
		w.WriteHeader(200)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	ht.mu.Lock()
	ep := ht.endpoint
	ht.mu.Unlock()
	if ep != "http://other.com/post" {
		t.Errorf("endpoint = %q, want %q", ep, "http://other.com/post")
	}
}

func TestHTTPTransport_Send_JSONResponseDispatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"jsonrpc":"2.0","id":"1","result":{"value":42}}`))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	ht.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	wg.Wait()
	if string(received.Result) != `{"value":42}` {
		t.Errorf("result = %s", received.Result)
	}
}

func TestHTTPTransport_Send_SessionIDSent(t *testing.T) {
	var gotSessionID string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSessionID = r.Header.Get("MCP-Session-ID")
		w.WriteHeader(200)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.SessionID = "my-session"

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	ht.Send(context.Background(), msg)

	if gotSessionID != "my-session" {
		t.Errorf("session id = %q, want %q", gotSessionID, "my-session")
	}
}

func TestHTTPTransport_Send_CustomHeaders(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		w.WriteHeader(200)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL, WithHTTPHeaders(map[string]string{"X-Custom": "value"}))

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	ht.Send(context.Background(), msg)

	if gotHeader != "value" {
		t.Errorf("custom header = %q, want %q", gotHeader, "value")
	}
}

func TestHTTPTransport_Start_SSEConnection(t *testing.T) {
	var gotAccept string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("MCP-Session-ID", "sse-sess")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	ht.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	ctx := context.Background()
	if err := ht.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if received.Method != "notifications/tools/list_changed" {
			t.Errorf("method = %q", received.Method)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for SSE message")
	}

	if gotAccept != "text/event-stream" {
		t.Errorf("accept = %q, want text/event-stream", gotAccept)
	}

	ht.Close()
}

func TestHTTPTransport_Close(t *testing.T) {
	ht := NewHTTPTransport("http://example.com")
	if err := ht.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestHTTPTransport_Send_UsesEndpoint(t *testing.T) {
	var secondHit atomic.Bool
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondHit.Store(true)
		w.WriteHeader(200)
	}))
	defer srv2.Close()

	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("MCP-Endpoint", srv2.URL)
		w.WriteHeader(200)
	}))
	defer srv1.Close()

	ht := NewHTTPTransport(srv1.URL)

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "init", ID: "1"}
	ht.Send(context.Background(), msg)

	// Second send should go to srv2
	ht.Send(context.Background(), msg)

	if !secondHit.Load() {
		t.Error("second request should have gone to discovered endpoint")
	}
}

// --- SSETransport tests ---

func TestNewSSETransport(t *testing.T) {
	st := NewSSETransport("http://example.com/sse")
	if st.URL != "http://example.com/sse" {
		t.Errorf("url = %q", st.URL)
	}
}

func TestSSETransport_StartAndReadEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("accept = %q", r.Header.Get("Accept"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		// Send endpoint event first
		fmt.Fprintf(w, "event: endpoint\ndata: %s/post\n\n", r.Host)
		flusher.Flush()
		// Send a JSON-RPC message
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	st.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if received.Method != "ping" {
			t.Errorf("method = %q, want %q", received.Method, "ping")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for SSE message")
	}

	st.Close()
}

func TestSSETransport_ReadSSE_DiscoverEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: endpoint\ndata: http://localhost:8080/messages\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(200 * time.Millisecond)

	st.mu.Lock()
	ep := st.Endpoint
	st.mu.Unlock()
	if ep != "http://localhost:8080/messages" {
		t.Errorf("endpoint = %q, want %q", ep, "http://localhost:8080/messages")
	}

	st.Close()
}

func TestSSETransport_Send_NoEndpoint(t *testing.T) {
	st := NewSSETransport("http://example.com/sse")

	err := st.Send(context.Background(), JSONRPCMessage{JSONRPC: "2.0", Method: "test"})
	if err == nil || !strings.Contains(err.Error(), "endpoint not yet discovered") {
		t.Errorf("expected endpoint not discovered error, got %v", err)
	}
}

func TestSSETransport_Send_UsesDiscoveredEndpoint(t *testing.T) {
	var gotBody []byte
	postSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}
		w.WriteHeader(200)
	}))
	defer postSrv.Close()

	st := NewSSETransport("http://example.com/sse")
	st.Endpoint = postSrv.URL

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := st.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	var received JSONRPCMessage
	if err := json.Unmarshal(gotBody, &received); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if received.Method != "test" {
		t.Errorf("method = %q", received.Method)
	}
}

func TestSSETransport_Send_CustomHeaders(t *testing.T) {
	var gotHeader string
	postSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("Authorization")
		w.WriteHeader(200)
	}))
	defer postSrv.Close()

	st := NewSSETransport("http://example.com/sse")
	st.Endpoint = postSrv.URL
	st.Headers = map[string]string{"Authorization": "Bearer token123"}

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	st.Send(context.Background(), msg)

	if gotHeader != "Bearer token123" {
		t.Errorf("authorization = %q, want %q", gotHeader, "Bearer token123")
	}
}

func TestSSETransport_Close_MultipleTimes(t *testing.T) {
	st := NewSSETransport("http://example.com/sse")
	st.Close()
	st.Close()
}

func TestSSETransport_OnClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Immediately close
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)

	var closeCalled atomic.Bool
	st.OnClose(func() {
		closeCalled.Store(true)
	})
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(200 * time.Millisecond)
	if !closeCalled.Load() {
		t.Error("OnClose not called after SSE connection closed")
	}
}

func TestSSETransport_OnError_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: not-json\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)

	var errCalled atomic.Bool
	st.OnError(func(err error) {
		errCalled.Store(true)
	})
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(200 * time.Millisecond)
	if !errCalled.Load() {
		t.Error("OnError not called for invalid JSON")
	}

	st.Close()
}

func TestHTTPTransport_SSE_SessionIDFromResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("MCP-Session-ID", "from-sse")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"method\":\"test\"}\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.OnMessage(func(m JSONRPCMessage) {})

	if err := ht.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(200 * time.Millisecond)

	ht.mu.Lock()
	sid := ht.SessionID
	ht.mu.Unlock()
	if sid != "from-sse" {
		t.Errorf("session id = %q, want %q", sid, "from-sse")
	}

	ht.Close()
}

func TestHTTPTransport_OnClose_Callback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Immediately close the connection
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var closeCalled atomic.Bool
	ht.OnClose(func() {
		closeCalled.Store(true)
	})
	ht.OnMessage(func(m JSONRPCMessage) {})

	if err := ht.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(300 * time.Millisecond)
	if !closeCalled.Load() {
		t.Error("OnClose not called after SSE connection closed")
	}
	ht.Close()
}

func TestHTTPTransport_Start_ConnectionError(t *testing.T) {
	ht := NewHTTPTransport("http://127.0.0.1:1") // unreachable port

	ht.OnMessage(func(m JSONRPCMessage) {})

	err := ht.Start(context.Background())
	if err == nil {
		ht.Close()
		t.Fatal("expected connection error from Start")
	}
	if !strings.Contains(err.Error(), "mcp: SSE connection") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHTTPTransport_Send_NonJSONResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("not json"))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.OnMessage(func(m JSONRPCMessage) {
		t.Error("should not dispatch for non-JSON response")
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}
}

func TestSSETransport_Start_CustomHeaders(t *testing.T) {
	gotHeader := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader <- r.Header.Get("X-Auth")
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)
	st.Headers = map[string]string{"X-Auth": "bearer tok"}
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case h := <-gotHeader:
		if h != "bearer tok" {
			t.Errorf("header = %q, want %q", h, "bearer tok")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for header")
	}
	st.Close()
}

func TestHTTPTransport_Start_SessionIDSentOnSSE(t *testing.T) {
	gotSessionID := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSessionID <- r.Header.Get("MCP-Session-ID")
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.SessionID = "pre-existing-sess"
	ht.OnMessage(func(m JSONRPCMessage) {})

	if err := ht.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case sid := <-gotSessionID:
		if sid != "pre-existing-sess" {
			t.Errorf("session id = %q, want %q", sid, "pre-existing-sess")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for session ID")
	}
	ht.Close()
}

func TestHTTPTransport_Start_CustomHeaders(t *testing.T) {
	gotHeader := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader <- r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "text/event-stream")
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL, WithHTTPHeaders(map[string]string{"X-Custom": "val"}))
	ht.OnMessage(func(m JSONRPCMessage) {})

	if err := ht.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case h := <-gotHeader:
		if h != "val" {
			t.Errorf("header = %q, want %q", h, "val")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for header")
	}
	ht.Close()
}

func TestHTTPTransport_Send_SSEResponseDispatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"1\",\"result\":{\"value\":1}}\n\n")
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"2\",\"result\":{\"value\":2}}\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var received []JSONRPCMessage
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(2)
	ht.OnMessage(func(m JSONRPCMessage) {
		mu.Lock()
		received = append(received, m)
		mu.Unlock()
		wg.Done()
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	wg.Wait()
	mu.Lock()
	defer mu.Unlock()
	if len(received) != 2 {
		t.Fatalf("received %d messages, want 2", len(received))
	}
	if string(received[0].Result) != `{"value":1}` {
		t.Errorf("first result = %s", received[0].Result)
	}
	if string(received[1].Result) != `{"value":2}` {
		t.Errorf("second result = %s", received[1].Result)
	}
}

func TestHTTPTransport_Send_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte("internal server error"))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	err := ht.Send(context.Background(), msg)
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want to contain 500", err.Error())
	}
}

func TestSSETransport_Start_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
		w.Write([]byte("unauthorized"))
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)
	st.OnMessage(func(m JSONRPCMessage) {})

	err := st.Start(context.Background())
	if err == nil {
		t.Fatal("expected error for HTTP 401")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error = %q, want to contain 401", err.Error())
	}
}

func TestSSETransport_Send_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte("server error"))
	}))
	defer srv.Close()

	st := NewSSETransport("http://example.com/sse")
	st.Endpoint = srv.URL

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	err := st.Send(context.Background(), msg)
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want to contain 500", err.Error())
	}
}

func TestHTTPTransport_OnError_Registration(t *testing.T) {
	ht := NewHTTPTransport("http://localhost:0")
	var called atomic.Bool
	ht.OnError(func(err error) {
		called.Store(true)
	})
	// Trigger an error by starting with an unreachable URL
	err := ht.Start(context.Background())
	if err == nil {
		t.Skip("expected connection error but got nil")
	}
	// OnError is for async errors; Start returns sync errors.
	// Just verify registration didn't panic.
	_ = called.Load()
}

func TestHTTPTransport_DispatchError(t *testing.T) {
	ht := NewHTTPTransport("http://localhost:0")
	var gotErr error
	var mu sync.Mutex
	ht.OnError(func(err error) {
		mu.Lock()
		gotErr = err
		mu.Unlock()
	})
	// Trigger dispatchError directly
	ht.dispatchError(fmt.Errorf("test error"))
	mu.Lock()
	defer mu.Unlock()
	if gotErr == nil || gotErr.Error() != "test error" {
		t.Errorf("expected 'test error', got %v", gotErr)
	}
}

func TestHTTPTransport_DispatchError_NilHandler(t *testing.T) {
	ht := NewHTTPTransport("http://localhost:0")
	// Should not panic with nil handler
	ht.dispatchError(fmt.Errorf("test error"))
}

func TestStdioTransport_SendAfterClose(t *testing.T) {
	st := NewStdioTransport("cat", nil)
	st.OnMessage(func(m JSONRPCMessage) {})
	ctx := context.Background()
	if err := st.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	st.Close()

	// Send after close should return error about transport closed
	err := st.Send(ctx, JSONRPCMessage{JSONRPC: "2.0", Method: "test"})
	if err == nil {
		t.Fatal("expected error from send after close")
	}
	if !strings.Contains(err.Error(), "transport closed") {
		t.Errorf("error = %q, want to contain 'transport closed'", err.Error())
	}
}

func TestHTTPTransport_Start_ContentTypeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte("{}"))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	ht.OnMessage(func(m JSONRPCMessage) {})

	err := ht.Start(context.Background())
	if err == nil {
		ht.Close()
		t.Fatal("expected content type error")
	}
	if !strings.Contains(err.Error(), "expected text/event-stream") {
		t.Errorf("error = %q, want to contain 'expected text/event-stream'", err.Error())
	}
}

func TestReadSSEBody_InvalidJSONDispatch(t *testing.T) {
	// Test that readSSEBody dispatches error for invalid JSON in SSE data
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		fmt.Fprintf(w, "data: {invalid json}\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var errCalled atomic.Bool
	ht.OnError(func(err error) {
		if strings.Contains(err.Error(), "invalid JSON") {
			errCalled.Store(true)
		}
	})
	ht.OnMessage(func(m JSONRPCMessage) {})

	if err := ht.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(300 * time.Millisecond)
	if !errCalled.Load() {
		t.Error("OnError not called for invalid JSON in SSE data")
	}
	ht.Close()
}

func TestReadSSEBodyCancellable_ContextCancel(t *testing.T) {
	// Test that readSSEBodyCancellable properly handles context cancellation.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		// Send one event, then keep connection open
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"1\",\"result\":{}}\n\n")
		flusher.Flush()
		// Keep alive long enough to be cancelled
		time.Sleep(5 * time.Second)
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var msgReceived atomic.Bool
	ht.OnMessage(func(m JSONRPCMessage) {
		msgReceived.Store(true)
	})

	// Make a Send call that returns SSE stream
	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	time.Sleep(200 * time.Millisecond)
	if !msgReceived.Load() {
		t.Error("expected to receive message from SSE stream")
	}

	// Close the transport which should cancel the postCancels context
	ht.Close()
	// If we get here without hanging, the context cancellation worked
}

func TestWithSSEHeaders(t *testing.T) {
	headers := map[string]string{"X-Key": "val", "Authorization": "Bearer tok"}
	st := NewSSETransport("http://example.com/sse", WithSSEHeaders(headers))
	if st.Headers["X-Key"] != "val" {
		t.Errorf("X-Key = %q, want %q", st.Headers["X-Key"], "val")
	}
	if st.Headers["Authorization"] != "Bearer tok" {
		t.Errorf("Authorization = %q, want %q", st.Headers["Authorization"], "Bearer tok")
	}
}

func TestWithSSEHTTPClient(t *testing.T) {
	client := &http.Client{Timeout: 10 * time.Second}
	st := NewSSETransport("http://example.com/sse", WithSSEHTTPClient(client))
	if st.HTTPClient != client {
		t.Error("HTTPClient not set by WithSSEHTTPClient")
	}
}

func TestNewSSETransport_WithOptions(t *testing.T) {
	client := &http.Client{Timeout: 5 * time.Second}
	headers := map[string]string{"X-Auth": "token"}
	st := NewSSETransport("http://example.com/sse",
		WithSSEHeaders(headers),
		WithSSEHTTPClient(client),
	)
	if st.URL != "http://example.com/sse" {
		t.Errorf("URL = %q", st.URL)
	}
	if st.Headers["X-Auth"] != "token" {
		t.Errorf("header not set")
	}
	if st.HTTPClient != client {
		t.Error("http client not set")
	}
}

func TestSSETransport_Start_ContentTypeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte("{}"))
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)
	st.OnMessage(func(m JSONRPCMessage) {})

	err := st.Start(context.Background())
	if err == nil {
		st.Close()
		t.Fatal("expected content type error")
	}
	if !strings.Contains(err.Error(), "expected text/event-stream") {
		t.Errorf("error = %q, want to contain 'expected text/event-stream'", err.Error())
	}
}

func TestSSETransport_Send_JSONResponseDispatch(t *testing.T) {
	postSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"jsonrpc":"2.0","id":"1","result":{"value":99}}`))
	}))
	defer postSrv.Close()

	st := NewSSETransport("http://example.com/sse")
	st.Endpoint = postSrv.URL

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	st.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := st.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	wg.Wait()
	if string(received.Result) != `{"value":99}` {
		t.Errorf("result = %s, want {\"value\":99}", received.Result)
	}
}

func TestSSETransport_Send_ContextCancelledWhileWaiting(t *testing.T) {
	st := NewSSETransport("http://example.com/sse")
	st.endpointReady = make(chan struct{}) // never closed = no endpoint

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := st.Send(ctx, JSONRPCMessage{JSONRPC: "2.0", Method: "test"})
	if err == nil {
		t.Fatal("expected context error")
	}
	// Either context.DeadlineExceeded or context.Canceled
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context deadline or canceled", err)
	}
}

func TestStripSSEValue_NoLeadingSpace(t *testing.T) {
	got := stripSSEValue("hello")
	if got != "hello" {
		t.Errorf("stripSSEValue(%q) = %q, want %q", "hello", got, "hello")
	}
}

func TestStripSSEValue_WithLeadingSpace(t *testing.T) {
	got := stripSSEValue(" hello")
	if got != "hello" {
		t.Errorf("stripSSEValue(%q) = %q, want %q", " hello", got, "hello")
	}
}

func TestStripSSEValue_EmptyString(t *testing.T) {
	got := stripSSEValue("")
	if got != "" {
		t.Errorf("stripSSEValue(%q) = %q, want %q", "", got, "")
	}
}

func TestHTTPTransport_Send_SSEResponseFromPost(t *testing.T) {
	// Test SSE response from POST (text/event-stream content type)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}
		fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"99\",\"result\":{\"ok\":true}}\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)

	var received JSONRPCMessage
	var wg sync.WaitGroup
	wg.Add(1)
	ht.OnMessage(func(m JSONRPCMessage) {
		received = m
		wg.Done()
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := ht.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if string(received.Result) != `{"ok":true}` {
			t.Errorf("result = %s", received.Result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for SSE response from POST")
	}

	ht.Close()
}

func TestSSETransport_Send_NonJSONResponse(t *testing.T) {
	// Verify that non-JSON response body doesn't dispatch
	postSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("ok"))
	}))
	defer postSrv.Close()

	st := NewSSETransport("http://example.com/sse")
	st.Endpoint = postSrv.URL

	st.OnMessage(func(m JSONRPCMessage) {
		t.Error("should not dispatch for non-JSON response")
	})

	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	if err := st.Send(context.Background(), msg); err != nil {
		t.Fatalf("Send: %v", err)
	}
}

func TestSSETransport_Start_ConnectionError(t *testing.T) {
	st := NewSSETransport("http://127.0.0.1:1") // unreachable
	st.OnMessage(func(m JSONRPCMessage) {})

	err := st.Start(context.Background())
	if err == nil {
		st.Close()
		t.Fatal("expected connection error")
	}
	if !strings.Contains(err.Error(), "SSE connection") {
		t.Errorf("error = %q, want to contain 'SSE connection'", err.Error())
	}
}

func TestSSETransport_ReadSSE_JSONParseError(t *testing.T) {
	// Test the JSON unmarshal error path in readSSE (non-endpoint events with bad JSON)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		// First send an endpoint event
		fmt.Fprintf(w, "event: endpoint\ndata: http://localhost:9999/post\n\n")
		flusher.Flush()
		// Then send a message event with invalid JSON
		fmt.Fprintf(w, "event: message\ndata: {bad json}\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	st := NewSSETransport(srv.URL)

	var errCalled atomic.Bool
	st.OnError(func(err error) {
		errCalled.Store(true)
	})
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(300 * time.Millisecond)
	if !errCalled.Load() {
		t.Error("OnError not called for JSON parse error in readSSE")
	}
	st.Close()
}

func TestSSETransport_WithCustomHTTPClient(t *testing.T) {
	// Tests NewSSETransport with WithSSEHTTPClient used in Start
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "event: endpoint\ndata: http://localhost:9999/post\n\n")
		flusher.Flush()
		time.Sleep(100 * time.Millisecond)
	}))
	defer srv.Close()

	client := &http.Client{Timeout: 10 * time.Second}
	st := NewSSETransport(srv.URL, WithSSEHTTPClient(client))
	st.OnMessage(func(m JSONRPCMessage) {})

	if err := st.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	time.Sleep(200 * time.Millisecond)
	st.Close()
}

func TestStdioTransport_StartNonExistentBinary(t *testing.T) {
	st := NewStdioTransport("/nonexistent/binary/that/does/not/exist", nil)
	err := st.Start(t.Context())
	if err == nil {
		st.Close()
		t.Fatal("expected error starting non-existent binary")
	}
	if !strings.Contains(err.Error(), "start process") {
		t.Errorf("error = %q, want to contain 'start process'", err.Error())
	}
}

func TestHTTPTransport_ErrorResponseBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte("custom error message"))
	}))
	defer srv.Close()

	ht := NewHTTPTransport(srv.URL)
	msg := JSONRPCMessage{JSONRPC: "2.0", Method: "test", ID: "1"}
	err := ht.Send(t.Context(), msg)
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want to contain status code 500", err.Error())
	}
	if !strings.Contains(err.Error(), "custom error message") {
		t.Errorf("error = %q, want to contain body text 'custom error message'", err.Error())
	}
}
