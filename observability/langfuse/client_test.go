package langfuse

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewID(t *testing.T) {
	id := newID()
	if len(id) != 36 {
		t.Errorf("newID() length = %d, want 36", len(id))
	}
	parts := strings.Split(id, "-")
	if len(parts) != 5 {
		t.Errorf("newID() parts = %d, want 5 dash-separated parts; got %q", len(parts), id)
	}
	// UUID version bit: character at position 14 must be '4'
	if id[14] != '4' {
		t.Errorf("newID() version bit at position 14 = %q, want '4'; id=%q", string(id[14]), id)
	}
}

func TestFormatTime_Zero(t *testing.T) {
	got := formatTime(time.Time{})
	if got != "" {
		t.Errorf("formatTime(zero) = %q, want empty string", got)
	}
}

func TestFormatTime_NonZero(t *testing.T) {
	ts := time.Date(2024, 6, 15, 10, 30, 45, 123000000, time.UTC)
	got := formatTime(ts)
	if !strings.Contains(got, "2024-06-15") {
		t.Errorf("formatTime() = %q, want to contain '2024-06-15'", got)
	}
	if !strings.Contains(got, "10:30:45") {
		t.Errorf("formatTime() = %q, want to contain '10:30:45'", got)
	}
}

func TestFlush_Empty(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newClient(srv.URL, "pub", "sec")
	err := c.flush(t.Context())
	if err != nil {
		t.Fatalf("flush(empty) returned error: %v", err)
	}
	if called {
		t.Error("flush(empty) should not make any HTTP request")
	}
}

func TestFlush_SendsBatch(t *testing.T) {
	var gotAuth string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newClient(srv.URL, "pub", "sec")
	c.appendEvents([]ingestionEvent{
		{ID: "e1", Type: eventTrace, Timestamp: "2024-01-01T00:00:00.000Z", Body: map[string]any{"id": "t1"}},
	})
	if err := c.flush(t.Context()); err != nil {
		t.Fatalf("flush: %v", err)
	}

	if !strings.HasPrefix(gotAuth, "Basic ") {
		t.Errorf("Authorization header = %q, want Basic prefix", gotAuth)
	}
	batch, ok := gotBody["batch"].([]any)
	if !ok || len(batch) == 0 {
		t.Errorf("expected non-empty batch in body, got %v", gotBody)
	}
}

func TestFlush_ClearsAfterSend(t *testing.T) {
	reqCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqCount++
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newClient(srv.URL, "pub", "sec")
	c.appendEvents([]ingestionEvent{
		{ID: "e1", Type: eventTrace, Timestamp: "2024-01-01T00:00:00.000Z", Body: nil},
	})
	if err := c.flush(t.Context()); err != nil {
		t.Fatalf("first flush: %v", err)
	}
	// second flush should be no-op
	if err := c.flush(t.Context()); err != nil {
		t.Fatalf("second flush: %v", err)
	}

	if reqCount != 1 {
		t.Errorf("HTTP requests = %d, want 1 (second flush should be no-op)", reqCount)
	}
}

func TestFlush_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := newClient(srv.URL, "bad-pub", "bad-sec")
	c.appendEvents([]ingestionEvent{
		{ID: "e1", Type: eventTrace, Timestamp: "2024-01-01T00:00:00.000Z", Body: nil},
	})
	err := c.flush(context.Background())
	if err == nil {
		t.Error("flush with 401 response should return error")
	}
	if err != nil && strings.HasSuffix(err.Error(), ": ") {
		t.Errorf("an empty response body must not leave a dangling separator: %q", err)
	}
}

// respondWith serves every ingestion request with the given status and body,
// counting requests.
func respondWith(t *testing.T, status int, body string) (*httptest.Server, *int32) {
	t.Helper()
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

// partial207 is the shape Langfuse returns for a batch of one valid and one
// invalid event (error detail shortened).
const partial207 = `{"successes":[{"id":"ok-1","status":201}],"errors":[{"id":"bad-1","status":400,"message":"Invalid request data","error":"[\n {\"code\": \"invalid_format\", \"path\": [\"timestamp\"], \"message\": \"Invalid ISO datetime\"}\n]"}]}`

func TestFlush_ReportsEventsRejectedInsideA207(t *testing.T) {
	srv, calls := respondWith(t, http.StatusMultiStatus, partial207)
	c := newClient(srv.URL, "pub", "sec")
	c.appendEvents([]ingestionEvent{{ID: "ok-1", Type: eventTrace}, {ID: "bad-1", Type: eventSpan}})

	err := c.flush(t.Context())
	var rej *IngestionRejectedError
	if !errors.As(err, &rej) {
		t.Fatalf("flush error = %v, want *IngestionRejectedError", err)
	}
	if rej.Accepted != 1 || len(rej.Rejected) != 1 || rej.Rejected[0].ID != "bad-1" || rej.Rejected[0].Status != 400 {
		t.Errorf("rejection = %+v", rej)
	}
	msg := err.Error()
	for _, want := range []string{"rejected 1 of 2 events", "bad-1", "status 400", "Invalid request data", "Invalid ISO datetime"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
	if strings.Contains(msg, "\n") {
		t.Errorf("error must be a single line: %q", msg)
	}
	if n := atomic.LoadInt32(calls); n != 1 {
		t.Errorf("HTTP requests = %d, want 1 (rejected events are not resent)", n)
	}
}

func TestFlush_207WithOnlySuccesses(t *testing.T) {
	srv, _ := respondWith(t, http.StatusMultiStatus, `{"successes":[{"id":"e1","status":201}],"errors":[]}`)
	c := newClient(srv.URL, "pub", "sec")
	c.appendEvents([]ingestionEvent{{ID: "e1", Type: eventTrace}})
	if err := c.flush(t.Context()); err != nil {
		t.Fatalf("flush: %v", err)
	}
}

// A body that is empty or not the documented shape says nothing about
// rejection; the status already said the batch was accepted.
func TestFlush_UnrecognizedSuccessBody(t *testing.T) {
	for _, body := range []string{"", "not json", `{"unexpected":true}`} {
		srv, _ := respondWith(t, http.StatusMultiStatus, body)
		c := newClient(srv.URL, "pub", "sec")
		c.appendEvents([]ingestionEvent{{ID: "e1", Type: eventTrace}})
		if err := c.flush(t.Context()); err != nil {
			t.Errorf("body %q: flush: %v", body, err)
		}
	}
}

func TestFlush_HTTPErrorIncludesResponseBody(t *testing.T) {
	srv, _ := respondWith(t, http.StatusUnauthorized, `{"message":"Invalid public key"}`)
	c := newClient(srv.URL, "pub", "sec")
	c.appendEvents([]ingestionEvent{{ID: "e1", Type: eventTrace}})
	err := c.flush(t.Context())
	if err == nil || !strings.Contains(err.Error(), "status 401") || !strings.Contains(err.Error(), "Invalid public key") {
		t.Fatalf("flush error = %v, want status and response body", err)
	}
}

func TestIngestionRejectedError_CapsListing(t *testing.T) {
	e := &IngestionRejectedError{Accepted: 2, Rejected: []RejectedEvent{
		{ID: "e1", Status: 400}, {ID: "e2", Status: 400}, {ID: "e3", Status: 400}, {ID: "e4", Status: 400}, {ID: "e5", Status: 400},
	}}
	msg := e.Error()
	if !strings.Contains(msg, "rejected 5 of 7 events") || !strings.Contains(msg, "e3") || strings.Contains(msg, "e4") || !strings.Contains(msg, "and 2 more") {
		t.Errorf("Error() = %q", msg)
	}
}

func TestErrorDetail(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"nil", nil, ""},
		{"string collapses whitespace", "a\n  b", "a b"},
		{"structured", map[string]any{"code": "invalid_type"}, `{"code":"invalid_type"}`},
		{"unmarshalable", math.Inf(1), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := errorDetail(tc.in); got != tc.want {
				t.Errorf("errorDetail(%v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestSnippet(t *testing.T) {
	if got := snippet("  short  text ", 100); got != "short text" {
		t.Errorf("snippet(short) = %q", got)
	}
	if got := snippet(strings.Repeat("é", 10), 5); got != "éé…" {
		t.Errorf("snippet must cut on a rune boundary, got %q", got)
	}
}
