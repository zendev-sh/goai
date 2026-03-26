package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// Transport defines the interface for MCP client-server communication.
// Implementations handle the underlying protocol (stdio, HTTP, SSE).
type Transport interface {
	// Start initializes the transport and begins reading messages.
	Start(ctx context.Context) error

	// Send transmits a JSON-RPC message to the server.
	Send(ctx context.Context, msg JSONRPCMessage) error

	// Close shuts down the transport and releases resources.
	Close() error

	// OnMessage registers a callback for incoming messages.
	OnMessage(func(JSONRPCMessage))

	// OnClose registers a callback invoked when the transport closes.
	OnClose(func())

	// OnError registers a callback for transport-level errors.
	OnError(func(error))
}

// StdioOption configures a StdioTransport.
type StdioOption func(*StdioTransport)

// WithStdioEnv sets environment variables for the subprocess.
func WithStdioEnv(env map[string]string) StdioOption {
	return func(t *StdioTransport) { t.Env = env }
}

// WithStdioDir sets the working directory for the subprocess.
func WithStdioDir(dir string) StdioOption {
	return func(t *StdioTransport) { t.Dir = dir }
}

// WithStdioStderr sets the writer for subprocess stderr output.
func WithStdioStderr(w io.Writer) StdioOption {
	return func(t *StdioTransport) { t.Stderr = w }
}

// StdioTransport communicates with a local MCP server via stdin/stdout
// of a child process.
type StdioTransport struct {
	// Command is the executable to run.
	Command string

	// Args are the command-line arguments.
	Args []string

	// Env contains additional environment variables for the subprocess.
	Env map[string]string

	// Dir sets the working directory. Empty uses the current directory.
	Dir string

	// Stderr receives the subprocess's stderr output. Nil discards it.
	Stderr io.Writer

	cmd     *exec.Cmd
	stdin   io.WriteCloser
	writeMu sync.Mutex

	mu        sync.Mutex
	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)
}

// NewStdioTransport creates a StdioTransport for the given command and arguments.
func NewStdioTransport(command string, args []string, opts ...StdioOption) *StdioTransport {
	t := &StdioTransport{
		Command: command,
		Args:    args,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// OnMessage registers a callback invoked for each incoming JSON-RPC message.
func (t *StdioTransport) OnMessage(fn func(JSONRPCMessage)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onMessage = fn
}

// OnClose registers a callback invoked when the transport closes.
func (t *StdioTransport) OnClose(fn func()) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onClose = fn
}

// OnError registers a callback invoked on transport-level errors.
func (t *StdioTransport) OnError(fn func(error)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onError = fn
}

// Start spawns the subprocess and begins reading from its stdout.
func (t *StdioTransport) Start(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, t.Command, t.Args...)
	cmd.Dir = t.Dir
	cmd.Env = mergeEnv(t.Env)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("mcp: stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("mcp: stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("mcp: stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("mcp: start process: %w", err)
	}
	t.writeMu.Lock()
	t.stdin = stdin
	t.cmd = cmd
	t.writeMu.Unlock()

	go t.readLoop(stdout)

	if t.Stderr != nil {
		go io.Copy(t.Stderr, stderr) //nolint:errcheck
	} else {
		go io.Copy(io.Discard, stderr) //nolint:errcheck
	}

	return nil
}

func (t *StdioTransport) readLoop(r io.Reader) {
	decoder := json.NewDecoder(r)
	for {
		var msg JSONRPCMessage
		if err := decoder.Decode(&msg); err != nil {
			if errors.Is(err, io.EOF) {
				t.dispatchClose()
			} else {
				t.dispatchError(err)
			}
			return
		}
		t.dispatchMessage(msg)
	}
}

// Send writes a JSON-RPC message to the subprocess stdin.
func (t *StdioTransport) Send(_ context.Context, msg JSONRPCMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	if t.stdin == nil {
		return fmt.Errorf("mcp: transport closed")
	}
	_, err = t.stdin.Write(append(data, '\n'))
	return err
}

// Close terminates the subprocess. Safe to call multiple times.
func (t *StdioTransport) Close() error {
	t.writeMu.Lock()
	stdin := t.stdin
	t.stdin = nil
	cmd := t.cmd
	t.cmd = nil
	t.writeMu.Unlock()
	if stdin != nil {
		stdin.Close()
	}
	if cmd != nil && cmd.Process != nil {
		cmd.Process.Kill() //nolint:errcheck
		cmd.Wait()         //nolint:errcheck
	}
	return nil
}

func (t *StdioTransport) dispatchMessage(msg JSONRPCMessage) {
	t.mu.Lock()
	fn := t.onMessage
	t.mu.Unlock()
	if fn != nil {
		fn(msg)
	}
}

func (t *StdioTransport) dispatchError(err error) {
	t.mu.Lock()
	fn := t.onError
	t.mu.Unlock()
	if fn != nil {
		fn(err)
	}
}

func (t *StdioTransport) dispatchClose() {
	t.mu.Lock()
	fn := t.onClose
	t.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// mergeEnv combines additional environment variables with the current process env.
func mergeEnv(env map[string]string) []string {
	merged := os.Environ()
	for k, v := range env {
		merged = append(merged, k+"="+v)
	}
	return merged
}

// HTTPTransportOption configures an HTTPTransport.
type HTTPTransportOption func(*HTTPTransport)

// WithHTTPHeaders sets custom headers sent with every HTTP request.
func WithHTTPHeaders(headers map[string]string) HTTPTransportOption {
	return func(t *HTTPTransport) { t.Headers = headers }
}

// WithHTTPClient sets a custom HTTP client for the transport.
func WithHTTPClient(httpClient *http.Client) HTTPTransportOption {
	return func(t *HTTPTransport) { t.HTTPClient = httpClient }
}

// HTTPTransport implements Streamable HTTP transport (MCP 2025-03-26).
// It uses POST for client-to-server messages and GET SSE for server-to-client.
type HTTPTransport struct {
	// URL is the MCP server endpoint.
	URL string

	// Headers contains custom headers sent with every request.
	Headers map[string]string

	// HTTPClient is the HTTP client used for requests. Defaults to http.DefaultClient.
	HTTPClient *http.Client

	// SessionID is set by the server for session management.
	SessionID string

	endpoint    string
	cancel      context.CancelFunc
	postCancels []context.CancelFunc

	mu        sync.Mutex
	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)
}

// NewHTTPTransport creates an HTTPTransport for the given URL.
func NewHTTPTransport(url string, opts ...HTTPTransportOption) *HTTPTransport {
	t := &HTTPTransport{URL: url}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// OnMessage registers a callback invoked for each incoming JSON-RPC message.
func (t *HTTPTransport) OnMessage(fn func(JSONRPCMessage)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onMessage = fn
}

// OnClose registers a callback invoked when the SSE connection closes.
func (t *HTTPTransport) OnClose(fn func()) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onClose = fn
}

// OnError registers a callback invoked on transport-level errors.
func (t *HTTPTransport) OnError(fn func(error)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onError = fn
}

// Start initiates the SSE connection for server-to-client messages.
// The HTTP request is made synchronously so connection errors are returned
// directly. The SSE body read loop runs in a background goroutine.
func (t *HTTPTransport) Start(ctx context.Context) error {
	sseCtx, cancel := context.WithCancel(ctx)
	t.mu.Lock()
	t.cancel = cancel
	t.mu.Unlock()

	req, err := http.NewRequestWithContext(sseCtx, "GET", t.URL, nil)
	if err != nil {
		cancel()
		return fmt.Errorf("mcp: create SSE request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	t.mu.Lock()
	sid := t.SessionID
	headers := t.Headers
	t.mu.Unlock()
	if sid != "" {
		req.Header.Set("MCP-Session-ID", sid)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	client := t.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		return fmt.Errorf("mcp: SSE connection: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		cancel()
		return fmt.Errorf("mcp: SSE connection failed: HTTP %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		resp.Body.Close()
		cancel()
		return fmt.Errorf("mcp: expected text/event-stream, got %q", ct)
	}
	if id := resp.Header.Get("MCP-Session-ID"); id != "" {
		t.mu.Lock()
		t.SessionID = id
		t.mu.Unlock()
	}

	go t.readSSEBody(resp.Body)
	return nil
}

// readSSEBody reads SSE events from an open response body until EOF or error.
// It takes ownership of the body and closes it when done.
func (t *HTTPTransport) readSSEBody(body io.ReadCloser) {
	defer body.Close()

	reader := bufio.NewReader(body)
	for {
		_, data, err := readSSEEvent(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				t.dispatchClose()
			} else {
				t.dispatchError(err)
			}
			return
		}
		if data == "" {
			continue
		}
		msg, ok := parseSSEData(data)
		if !ok {
			t.dispatchError(fmt.Errorf("mcp: invalid JSON in SSE data: %s", data))
			continue
		}
		t.dispatchMessage(msg)
	}
}

// Send posts a JSON-RPC message to the server via HTTP POST.
func (t *HTTPTransport) Send(ctx context.Context, msg JSONRPCMessage) error {
	client := t.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	t.mu.Lock()
	postURL := t.endpoint
	sid := t.SessionID
	headers := t.Headers
	t.mu.Unlock()
	if postURL == "" {
		postURL = t.URL
	}

	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", postURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("mcp: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	if sid != "" {
		req.Header.Set("MCP-Session-ID", sid)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp: send request: %w", err)
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return fmt.Errorf("mcp: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	if endpoint := resp.Header.Get("MCP-Endpoint"); endpoint != "" {
		t.mu.Lock()
		t.endpoint = endpoint
		t.mu.Unlock()
	}
	if id := resp.Header.Get("MCP-Session-ID"); id != "" {
		t.mu.Lock()
		t.SessionID = id
		t.mu.Unlock()
	}

	// Read and dispatch any inline JSON-RPC responses from the POST body.
	// Some servers return results directly in the POST response.
	ct := resp.Header.Get("Content-Type")
	switch {
	case strings.HasPrefix(ct, "application/json"):
		var respMsg JSONRPCMessage
		if err := json.NewDecoder(resp.Body).Decode(&respMsg); err == nil && respMsg.JSONRPC != "" {
			t.dispatchMessage(respMsg)
		}
		resp.Body.Close()
	case strings.HasPrefix(ct, "text/event-stream"):
		// Read SSE events from POST response in a background goroutine
		// to avoid blocking the caller. Goroutine owns the body.
		postCtx, postCancel := context.WithCancel(context.Background())
		t.mu.Lock()
		t.postCancels = append(t.postCancels, postCancel)
		t.mu.Unlock()
		go t.readSSEBodyCancellable(postCtx, resp.Body)
	default:
		resp.Body.Close()
	}

	return nil
}

// readSSEBodyCancellable wraps readSSEBody with context cancellation support.
// When the context is cancelled, the body is closed to unblock the reader.
func (t *HTTPTransport) readSSEBodyCancellable(ctx context.Context, body io.ReadCloser) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		t.readSSEBody(body)
	}()
	select {
	case <-ctx.Done():
		body.Close() // force the reader to exit
		<-done        // wait for goroutine to finish
	case <-done:
		// natural exit
	}
}

// Close cancels the SSE goroutine and all POST SSE goroutines, then releases resources.
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	cancel := t.cancel
	t.cancel = nil
	cancels := t.postCancels
	t.postCancels = nil
	t.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	for _, cancel := range cancels {
		cancel()
	}
	return nil
}

func (t *HTTPTransport) dispatchMessage(msg JSONRPCMessage) {
	t.mu.Lock()
	fn := t.onMessage
	t.mu.Unlock()
	if fn != nil {
		fn(msg)
	}
}

func (t *HTTPTransport) dispatchError(err error) {
	t.mu.Lock()
	fn := t.onError
	t.mu.Unlock()
	if fn != nil {
		fn(err)
	}
}

func (t *HTTPTransport) dispatchClose() {
	t.mu.Lock()
	fn := t.onClose
	t.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// SSETransport implements the legacy SSE transport (MCP 2024-11-05).
// The server sends events via GET SSE, and the client sends requests via POST
// to an endpoint discovered from the SSE stream.
type SSETransport struct {
	// URL is the SSE endpoint URL.
	URL string

	// Headers contains custom headers sent with every request.
	Headers map[string]string

	// Endpoint is the POST endpoint discovered from the server's "endpoint" SSE event.
	Endpoint string

	// HTTPClient is the HTTP client used for requests. Defaults to http.DefaultClient.
	HTTPClient *http.Client

	respBody      io.ReadCloser
	endpointReady chan struct{} // closed when Endpoint is discovered

	mu        sync.Mutex
	onMessage func(JSONRPCMessage)
	onClose   func()
	onError   func(error)
}

// SSETransportOption configures an SSETransport.
type SSETransportOption func(*SSETransport)

// WithSSEHeaders sets custom headers sent with every SSE request.
func WithSSEHeaders(headers map[string]string) SSETransportOption {
	return func(t *SSETransport) { t.Headers = headers }
}

// WithSSEHTTPClient sets a custom HTTP client for the SSE transport.
func WithSSEHTTPClient(httpClient *http.Client) SSETransportOption {
	return func(t *SSETransport) { t.HTTPClient = httpClient }
}

// NewSSETransport creates a legacy SSETransport for the given URL.
func NewSSETransport(url string, opts ...SSETransportOption) *SSETransport {
	t := &SSETransport{URL: url}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// OnMessage registers a callback invoked for each incoming JSON-RPC message.
func (t *SSETransport) OnMessage(fn func(JSONRPCMessage)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onMessage = fn
}

// OnClose registers a callback invoked when the SSE connection closes.
func (t *SSETransport) OnClose(fn func()) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onClose = fn
}

// OnError registers a callback invoked on transport-level errors.
func (t *SSETransport) OnError(fn func(error)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onError = fn
}

// Start connects to the SSE endpoint and begins reading events.
// Start returns after the connection is established; it does NOT block for
// endpoint discovery. Send waits for the endpoint via endpointReady channel.
func (t *SSETransport) Start(ctx context.Context) error {
	t.endpointReady = make(chan struct{})
	client := t.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, "GET", t.URL, nil)
	if err != nil {
		return fmt.Errorf("mcp: create SSE request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	t.mu.Lock()
	headers := t.Headers
	t.mu.Unlock()
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp: SSE connection: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return fmt.Errorf("mcp: SSE connection failed: HTTP %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		resp.Body.Close()
		return fmt.Errorf("mcp: expected text/event-stream, got %q", ct)
	}
	t.respBody = resp.Body
	go t.readSSE(bufio.NewReader(resp.Body))
	return nil
}

func (t *SSETransport) readSSE(r *bufio.Reader) {
	for {
		eventType, data, err := readSSEEvent(r)
		if err != nil {
			if errors.Is(err, io.EOF) {
				t.dispatchClose()
			} else {
				t.dispatchError(err)
			}
			return
		}
		if data == "" {
			continue
		}

		// Legacy SSE transport: server sends "endpoint" event with POST URL
		if eventType == "endpoint" {
			t.mu.Lock()
			first := t.Endpoint == ""
			t.Endpoint = data
			t.mu.Unlock()
			if first && t.endpointReady != nil {
				close(t.endpointReady)
			}
			continue
		}

		var msg JSONRPCMessage
		if err := json.Unmarshal([]byte(data), &msg); err != nil {
			t.dispatchError(err)
			continue
		}
		t.dispatchMessage(msg)
	}
}

// Send posts a JSON-RPC message to the discovered endpoint.
// If the endpoint hasn't been discovered yet, Send waits for it or until ctx is cancelled.
func (t *SSETransport) Send(ctx context.Context, msg JSONRPCMessage) error {
	t.mu.Lock()
	endpoint := t.Endpoint
	t.mu.Unlock()
	if endpoint == "" {
		// Wait for endpoint discovery
		if t.endpointReady != nil {
			select {
			case <-t.endpointReady:
				t.mu.Lock()
				endpoint = t.Endpoint
				t.mu.Unlock()
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if endpoint == "" {
			return errors.New("mcp: SSE endpoint not yet discovered from server")
		}
	}

	client := t.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("mcp: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	t.mu.Lock()
	sseHeaders := t.Headers
	t.mu.Unlock()
	for k, v := range sseHeaders {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp: send request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("mcp: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	// Read and dispatch any JSON-RPC response from the POST body.
	ct := resp.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "application/json") {
		var respMsg JSONRPCMessage
		if err := json.NewDecoder(resp.Body).Decode(&respMsg); err == nil && respMsg.JSONRPC != "" {
			t.dispatchMessage(respMsg)
		}
	}
	return nil
}

// Close terminates the SSE connection. Safe to call multiple times.
func (t *SSETransport) Close() error {
	t.mu.Lock()
	body := t.respBody
	t.respBody = nil
	t.mu.Unlock()
	if body != nil {
		body.Close()
	}
	return nil
}

func (t *SSETransport) dispatchMessage(msg JSONRPCMessage) {
	t.mu.Lock()
	fn := t.onMessage
	t.mu.Unlock()
	if fn != nil {
		fn(msg)
	}
}

func (t *SSETransport) dispatchError(err error) {
	t.mu.Lock()
	fn := t.onError
	t.mu.Unlock()
	if fn != nil {
		fn(err)
	}
}

func (t *SSETransport) dispatchClose() {
	t.mu.Lock()
	fn := t.onClose
	t.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// readSSEEvent reads a complete SSE event (terminated by a blank line) from a
// bufio.Reader. It returns the event type, the accumulated data payload from all
// "data:" lines joined with newlines, and any read error. Per the SSE spec,
// multiple consecutive "data:" lines are concatenated with newlines.
func readSSEEvent(reader *bufio.Reader) (eventType string, data string, err error) {
	var dataLines []string
	for {
		line, readErr := reader.ReadString('\n')
		// Process the line even if readErr is set, since ReadString
		// returns data read before the error (e.g., EOF without newline).
		line = strings.TrimRight(line, "\r\n")
		if line != "" {
			if strings.HasPrefix(line, "event:") {
				eventType = stripSSEValue(strings.TrimPrefix(line, "event:"))
			} else if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, stripSSEValue(strings.TrimPrefix(line, "data:")))
			}
			// Ignore id:, retry:, and comment lines (: prefix)
		}
		if readErr != nil {
			if len(dataLines) > 0 {
				return eventType, strings.Join(dataLines, "\n"), nil
			}
			return "", "", readErr
		}
		if line == "" {
			// End of event (blank line)
			if len(dataLines) > 0 {
				return eventType, strings.Join(dataLines, "\n"), nil
			}
			continue // skip empty lines between events
		}
	}
}

// stripSSEValue strips a single leading space from an SSE field value per the
// SSE spec: "If value starts with a U+0020 SPACE character, remove it."
func stripSSEValue(s string) string {
	if strings.HasPrefix(s, " ") {
		return s[1:]
	}
	return s
}

// parseSSEData parses a JSON-RPC message from SSE data payload.
// Returns the parsed message and true if successful.
func parseSSEData(data string) (JSONRPCMessage, bool) {
	if data == "" {
		return JSONRPCMessage{}, false
	}
	var msg JSONRPCMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return JSONRPCMessage{}, false
	}
	return msg, true
}
