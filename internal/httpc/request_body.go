package httpc

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
)

// NewRequest builds a request with a replayable in-memory upload. Send it with Do.
func NewRequest(ctx context.Context, method, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err == nil && body != nil {
		SetBody(req, body)
	}
	return req, err
}

// SetBody installs an in-memory body before sending (including after rewriting
// a payload for signing). GetBody must replay exactly the bytes being sent.
func SetBody(req *http.Request, data []byte) {
	req.ContentLength = int64(len(data))
	if len(data) == 0 {
		req.Body = http.NoBody
		req.GetBody = func() (io.ReadCloser, error) { return http.NoBody, nil }
		return
	}
	// net/http can coalesce HTTP/1 headers and a known in-memory reader in
	// its default 4 KiB write buffer. Keep just a small snapshot for replay,
	// even if data is a subslice of a much larger allocation.
	if len(data) <= 4<<10 {
		snapshot := bytes.Clone(data)
		req.Body = io.NopCloser(bytes.NewReader(snapshot))
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(snapshot)), nil }
		return
	}
	body := &requestBody{remaining: data, replay: data}
	req.Body = body
	req.GetBody = body.getBody
}

// requestBody has two independent references to the same backing array.
// Reading the last byte (or Close) releases the upload reference; returning
// from Do with a standard transport releases replay, without interrupting an early response
// whose upload is still in progress. No request fields are mutated after send.
type requestBody struct {
	mu                sync.Mutex
	remaining, replay []byte
}

func (b *requestBody) Read(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.remaining) == 0 {
		return 0, io.EOF
	}
	n := copy(p, b.remaining)
	b.remaining = b.remaining[n:]
	if len(b.remaining) == 0 {
		b.remaining = nil
	}
	return n, nil
}

func (b *requestBody) Close() error {
	b.mu.Lock()
	b.remaining = nil
	b.mu.Unlock()
	return nil
}

func (b *requestBody) getBody() (io.ReadCloser, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.replay == nil {
		return nil, errors.New("httpc: request body is no longer available for replay")
	}
	return &requestBody{remaining: b.replay}, nil
}

func (b *requestBody) releaseReplay() {
	b.mu.Lock()
	b.replay = nil
	b.mu.Unlock()
}

// CanReleaseReplay reports whether the transport is done with GetBody when it
// returns. Unknown middleware may still call GetBody from response Body.Read.
func CanReleaseReplay(rt http.RoundTripper) bool {
	if rt == nil {
		rt = http.DefaultTransport
	}
	switch t := rt.(type) {
	case *http.Transport:
		return true
	case interface{ CanReleaseReplay() bool }:
		return t.CanReleaseReplay()
	default:
		return false
	}
}

// Do keeps the upload replayable across transport retries and client redirects.
func Do(client *http.Client, req *http.Request) (*http.Response, error) {
	if b, ok := req.Body.(*requestBody); ok && CanReleaseReplay(client.Transport) {
		defer b.releaseReplay()
	}
	return client.Do(req)
}

// DoRoundTrip applies the same lifetime to a rewritten request owned by an adapter.
func DoRoundTrip(transport http.RoundTripper, req *http.Request) (*http.Response, error) {
	if b, ok := req.Body.(*requestBody); ok && CanReleaseReplay(transport) {
		defer b.releaseReplay()
	}
	return transport.RoundTrip(req)
}
