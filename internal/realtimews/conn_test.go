package realtimews

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"go.uber.org/goleak"
)

func websocketURL(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http")
}

func TestDialReadWriteAndHeaders(t *testing.T) {
	defer goleak.VerifyNone(t)

	headerSeen := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headerSeen <- r.Header.Get("X-Test")
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{Subprotocols: []string{"realtime"}})
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()

		for i := 0; ; i++ {
			typ, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			if i < 2 {
				if err := conn.Write(r.Context(), typ, data); err != nil {
					return
				}
			}
		}
	}))
	defer server.Close()

	headers := http.Header{"X-Test": []string{"original"}}
	conn, resp, err := Dial(t.Context(), websocketURL(server.URL), Options{
		Headers:      headers,
		Subprotocols: []string{"realtime"},
		ReadLimit:    1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp == nil || resp.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("response = %#v", resp)
	}
	headers.Set("X-Test", "mutated")
	if got := <-headerSeen; got != "original" {
		t.Fatalf("header = %q", got)
	}

	for _, message := range []struct {
		typ  MessageType
		data []byte
	}{
		{MessageText, []byte(`{"hello":"world"}`)},
		{MessageBinary, []byte{1, 2, 3}},
	} {
		if err := conn.Write(t.Context(), message.typ, message.data); err != nil {
			t.Fatal(err)
		}
		typ, data, err := conn.Read(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if typ != message.typ || string(data) != string(message.data) {
			t.Fatalf("echo = (%v, %v), want (%v, %v)", typ, data, message.typ, message.data)
		}
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
}

func TestDialHandshakeFailureReturnsResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "denied", http.StatusUnauthorized)
	}))
	defer server.Close()

	conn, resp, err := Dial(t.Context(), websocketURL(server.URL), Options{})
	if err == nil || conn != nil {
		t.Fatalf("Dial = (%v, %#v), want failure", conn, err)
	}
	if resp == nil || resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("response = %#v", resp)
	}
	_ = resp.Body.Close()
}

func TestReadLimit(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		_ = conn.Write(r.Context(), websocket.MessageText, []byte("message-too-large"))
	}))
	defer server.Close()

	conn, _, err := Dial(t.Context(), websocketURL(server.URL), Options{ReadLimit: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.CloseNow() }()
	if _, _, err := conn.Read(t.Context()); err == nil {
		t.Fatal("expected read limit error")
	}
}

func TestReadCancellation(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		_, _, _ = conn.Read(context.Background())
	}))
	defer server.Close()

	conn, _, err := Dial(t.Context(), websocketURL(server.URL), Options{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, _, err := conn.Read(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Read error = %v", err)
	}
	_ = conn.CloseNow()
}

func TestConcurrentWrites(t *testing.T) {
	defer goleak.VerifyNone(t)

	const messages = 20
	received := make(chan string, messages)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		for range messages {
			_, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			received <- string(data)
		}
	}))
	defer server.Close()

	conn, _, err := Dial(t.Context(), websocketURL(server.URL), Options{})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, messages)
	for i := range messages {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs <- conn.Write(t.Context(), MessageText, []byte(fmt.Sprintf("message-%d", i)))
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Write error = %v", err)
		}
	}

	seen := make(map[string]bool, messages)
	for range messages {
		select {
		case message := <-received:
			seen[message] = true
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for messages")
		}
	}
	if len(seen) != messages {
		t.Fatalf("received %d unique messages", len(seen))
	}
	_ = conn.CloseNow()
}

func TestInvalidAndNilConnectionOperations(t *testing.T) {
	var conn *Conn
	if _, _, err := conn.Read(t.Context()); err == nil {
		t.Fatal("nil Read should fail")
	}
	if err := conn.Write(t.Context(), MessageText, nil); err == nil {
		t.Fatal("nil Write should fail")
	}
	if err := conn.Ping(t.Context()); err == nil {
		t.Fatal("nil Ping should fail")
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("nil Close = %v", err)
	}
	if err := conn.CloseNow(); err != nil {
		t.Fatalf("nil CloseNow = %v", err)
	}

	empty := &Conn{}
	if err := empty.Write(t.Context(), MessageType(99), nil); err == nil {
		t.Fatal("invalid Write should fail")
	}
}

func TestUnsupportedWriteType(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		_, _, _ = conn.Read(context.Background())
	}))
	defer server.Close()

	conn, _, err := Dial(t.Context(), websocketURL(server.URL), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := conn.Write(t.Context(), MessageType(99), nil); err == nil {
		t.Fatal("expected unsupported type error")
	}
	_ = conn.CloseNow()
}

func TestCloseStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		_ = conn.Close(websocket.StatusPolicyViolation, "quota exceeded")
	}))
	defer server.Close()

	conn, _, err := Dial(t.Context(), websocketURL(server.URL), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.CloseNow() }()

	_, _, err = conn.Read(t.Context())
	code, reason, ok := CloseStatus(err)
	if !ok || code != int(websocket.StatusPolicyViolation) || reason != "quota exceeded" {
		t.Fatalf("CloseStatus(%v) = %d, %q, %v", err, code, reason, ok)
	}
	if _, _, ok := CloseStatus(errors.New("plain")); ok {
		t.Fatal("plain error reported a close status")
	}
}
