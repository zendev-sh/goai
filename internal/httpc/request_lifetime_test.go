package httpc

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A retained marshaled upload is much larger than incidental HTTP allocations.
// Two GC cycles also evict the JSON encoder's temporary sync.Pool buffer.
func liveHeap() uint64 {
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func TestDoJSONRequest_ReleasesUploadDuringStream(t *testing.T) {
	const size = 8 << 20
	finish := make(chan struct{})
	endStream := sync.OnceFunc(func() { close(finish) })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read upload: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-finish
		fmt.Fprint(w, "data: end\n\n")
	}))
	defer srv.Close()
	defer endStream()

	before := liveHeap()
	resp, err := DoJSONRequest(t.Context(), RequestConfig{URL: srv.URL, HTTPClient: srv.Client(), Body: map[string]any{"data": strings.Repeat("x", size)}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if after := liveHeap(); after > before+size/2 {
		t.Errorf("open response retained %d additional heap bytes after %d-byte upload", after-before, size)
	}
	endStream()
	tail, err := io.ReadAll(resp.Body)
	if err != nil || string(tail) != "data: end\n\n" {
		t.Fatalf("stream tail = %q, err = %v", tail, err)
	}
	runtime.KeepAlive(resp)
}

func TestDoJSONRequest_ReplaysUploadOnRedirect(t *testing.T) {
	// Two hops exercise net/http's replay of the original request more than once.
	const want = `{"data":"redirect"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, err := io.ReadAll(r.Body)
		if err != nil || string(got) != want || r.Method != http.MethodPost || r.ContentLength != int64(len(want)) {
			t.Errorf("%s: method=%s length=%d body=%q err=%v", r.URL.Path, r.Method, r.ContentLength, got, err)
		}
		switch r.URL.Path {
		case "/":
			http.Redirect(w, r, "/again", http.StatusTemporaryRedirect)
		case "/again":
			http.Redirect(w, r, "/end", http.StatusPermanentRedirect)
		case "/end":
			fmt.Fprint(w, "done")
		}
	}))
	defer srv.Close()
	resp, err := DoJSONRequest(t.Context(), RequestConfig{URL: srv.URL, HTTPClient: srv.Client(), Body: map[string]any{"data": "redirect"}}, func(_ string, code int, _ []byte, _ http.Header) error { return fmt.Errorf("HTTP %d", code) })
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil || string(got) != "done" {
		t.Fatalf("final response = %q, err = %v", got, err)
	}
}

func TestDoJSONRequest_ReplaysUploadOnTransportRetry(t *testing.T) {
	var attempts atomic.Int32
	const want = `{"data":"retry"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		got, err := io.ReadAll(r.Body)
		if err != nil || r.Method != http.MethodPost || r.Header.Get("Idempotency-Key") != "retry-test" || string(got) != want {
			t.Errorf("%s key=%q body=%q err=%v", r.Method, r.Header.Get("Idempotency-Key"), got, err)
		}
		if attempts.Add(1) == 1 {
			// Close the reused connection without sending response headers.
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			conn.Close()
			return
		}
		fmt.Fprint(w, "retried")
	}))
	defer srv.Close()
	client := srv.Client()
	warm, err := client.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	warm.Body.Close() // The 204 has no body to drain before reuse.
	resp, err := DoJSONRequest(t.Context(), RequestConfig{
		URL: srv.URL, HTTPClient: client,
		Headers: map[string]string{"Idempotency-Key": "retry-test"},
		Body:    map[string]any{"data": "retry"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil || string(got) != "retried" || attempts.Load() != 2 {
		t.Fatalf("response=%q attempts=%d err=%v", got, attempts.Load(), err)
	}
}

func TestDoJSONRequest_EarlyResponsePreservesUpload(t *testing.T) {
	// Larger than the HTTP/2 stream window: the transport cannot finish
	// uploading until the handler starts consuming the request after Do returns.
	data := strings.Repeat("a", 4<<20)
	readUpload := make(chan struct{})
	resume := sync.OnceFunc(func() { close(readUpload) })
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		select {
		case <-readUpload:
		case <-r.Context().Done():
			return
		}
		got, err := io.ReadAll(r.Body)
		if err != nil || string(got) != `{"data":"`+data+`"}` {
			t.Errorf("upload after response headers: received %d bytes, err = %v", len(got), err)
		}
		fmt.Fprint(w, "done")
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()
	defer resume()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	resp, err := DoJSONRequest(ctx, RequestConfig{URL: srv.URL, HTTPClient: srv.Client(), Body: map[string]any{"data": data}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	resume()
	got, err := io.ReadAll(resp.Body)
	if err != nil || string(got) != "done" {
		t.Fatalf("final response = %q, err = %v", got, err)
	}
}
