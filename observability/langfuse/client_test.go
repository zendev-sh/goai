package langfuse

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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
}
