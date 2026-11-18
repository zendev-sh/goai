package httpc

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

type writeCountingConn struct {
	net.Conn
	writes *atomic.Int32
}

func (c writeCountingConn) Write(p []byte) (int, error) {
	c.writes.Add(1)
	return c.Conn.Write(p)
}

func TestSmallUploadUsesSingleHTTP1Write(t *testing.T) {
	const payload = `{"message":"hello"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != payload {
			t.Errorf("upload = %q, err = %v", body, err)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()
	var writes atomic.Int32
	transport := &http.Transport{DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		return writeCountingConn{conn, &writes}, nil
	}}
	defer transport.CloseIdleConnections()
	req, err := NewRequest(t.Context(), http.MethodPost, srv.URL, []byte(payload))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Do(&http.Client{Transport: transport}, req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := writes.Load(); got != 1 {
		t.Errorf("connection writes = %d; want 1 for headers and body together", got)
	}
}
