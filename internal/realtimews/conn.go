// Package realtimews provides the provider-neutral WebSocket transport used by
// realtime model adapters. It intentionally contains no OpenAI or Gemini wire
// protocol logic.
package realtimews

import (
	"context"
	"errors"
	"net/http"
	"sync"

	"github.com/coder/websocket"
)

// MessageType identifies a WebSocket message payload type.
type MessageType uint8

const (
	MessageText MessageType = iota + 1
	MessageBinary
)

// Options configures a WebSocket connection.
type Options struct {
	HTTPClient   *http.Client
	Headers      http.Header
	Subprotocols []string
	ReadLimit    int64
}

// Conn is a realtime WebSocket connection. Read must have a single caller;
// Write, Close, and Read may otherwise run concurrently.
type Conn struct {
	conn *websocket.Conn

	closeOnce sync.Once
	closeErr  error
}

// Dial opens a WebSocket connection. The HTTP response is returned when the
// handshake fails so provider adapters can decode API errors.
func Dial(ctx context.Context, url string, opts Options) (*Conn, *http.Response, error) {
	headers := cloneHeader(opts.Headers)
	subprotocols := append([]string(nil), opts.Subprotocols...)

	ws, resp, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		HTTPClient:   opts.HTTPClient,
		HTTPHeader:   headers,
		Subprotocols: subprotocols,
	})
	if err != nil {
		return nil, resp, err
	}
	if opts.ReadLimit > 0 {
		ws.SetReadLimit(opts.ReadLimit)
	}
	return &Conn{conn: ws}, resp, nil
}

// Read receives one complete text or binary message. The returned bytes are
// owned by the caller.
func (c *Conn) Read(ctx context.Context) (MessageType, []byte, error) {
	if c == nil || c.conn == nil {
		return 0, nil, errors.New("realtimews: nil connection")
	}
	typ, data, err := c.conn.Read(ctx)
	if err != nil {
		return 0, nil, err
	}
	switch typ {
	case websocket.MessageText:
		return MessageText, data, nil
	case websocket.MessageBinary:
		return MessageBinary, data, nil
	default:
		return 0, nil, errors.New("realtimews: unsupported message type")
	}
}

// Write sends one complete text or binary message. The payload is consumed
// before Write returns and is not retained.
func (c *Conn) Write(ctx context.Context, typ MessageType, data []byte) error {
	if c == nil || c.conn == nil {
		return errors.New("realtimews: nil connection")
	}
	var wsType websocket.MessageType
	switch typ {
	case MessageText:
		wsType = websocket.MessageText
	case MessageBinary:
		wsType = websocket.MessageBinary
	default:
		return errors.New("realtimews: unsupported message type")
	}
	return c.conn.Write(ctx, wsType, data)
}

// Ping verifies that the peer responds before ctx expires.
func (c *Conn) Ping(ctx context.Context) error {
	if c == nil || c.conn == nil {
		return errors.New("realtimews: nil connection")
	}
	return c.conn.Ping(ctx)
}

// Close performs a normal WebSocket close handshake. It is idempotent.
func (c *Conn) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	c.closeOnce.Do(func() {
		c.closeErr = c.conn.Close(websocket.StatusNormalClosure, "")
	})
	return c.closeErr
}

// CloseNow closes the underlying network connection immediately. It is useful
// when a protocol or backpressure error makes a close handshake undesirable.
func (c *Conn) CloseNow() error {
	if c == nil || c.conn == nil {
		return nil
	}
	c.closeOnce.Do(func() {
		c.closeErr = c.conn.CloseNow()
	})
	return c.closeErr
}

func cloneHeader(header http.Header) http.Header {
	if header == nil {
		return nil
	}
	clone := make(http.Header, len(header))
	for key, values := range header {
		clone[key] = append([]string(nil), values...)
	}
	return clone
}
