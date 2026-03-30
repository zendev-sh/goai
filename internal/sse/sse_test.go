package sse

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestScanner_BasicEvents(t *testing.T) {
	input := "data: hello\ndata: world\ndata: [DONE]\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "hello" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "hello")
	}

	data, ok = s.Next()
	if !ok || data != "world" {
		t.Errorf("second: got %q, %v; want %q, true", data, ok, "world")
	}

	data, ok = s.Next()
	if ok {
		t.Errorf("after DONE: got %q, %v; want false", data, ok)
	}
	if !s.IsDone() {
		t.Error("IsDone should be true after [DONE]")
	}
}

func TestScanner_SkipsNonDataLines(t *testing.T) {
	input := "event: message\nid: 1\ndata: payload\n\nretry: 5000\ndata: [DONE]\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "payload" {
		t.Errorf("got %q, %v; want %q, true", data, ok, "payload")
	}

	data, ok = s.Next()
	if ok {
		t.Errorf("after DONE: got %q, %v; want false", data, ok)
	}
}

func TestScanner_EmptyStream(t *testing.T) {
	s := NewScanner(strings.NewReader(""))

	data, ok := s.Next()
	if ok {
		t.Errorf("empty stream: got %q, %v; want false", data, ok)
	}
	if s.IsDone() {
		t.Error("IsDone should be false for empty stream (no [DONE] seen)")
	}
}

func TestScanner_NoDataPrefix(t *testing.T) {
	input := "event: ping\n\nevent: pong\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if ok {
		t.Errorf("no data lines: got %q, %v; want false", data, ok)
	}
}

func TestScanner_JSONPayloads(t *testing.T) {
	input := `data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}
data: {"id":"2","choices":[{"delta":{"content":" there"}}]}
data: [DONE]
`
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok {
		t.Fatal("expected first event")
	}
	if !strings.Contains(data, `"content":"hi"`) {
		t.Errorf("first event missing content: %s", data)
	}

	data, ok = s.Next()
	if !ok {
		t.Fatal("expected second event")
	}
	if !strings.Contains(data, `"content":" there"`) {
		t.Errorf("second event missing content: %s", data)
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected false after DONE")
	}
}

func TestScanner_DoneIdempotent(t *testing.T) {
	input := "data: first\ndata: [DONE]\ndata: after-done\n"
	s := NewScanner(strings.NewReader(input))

	s.Next() // "first"
	s.Next() // DONE

	// Calling Next after DONE should keep returning false.
	for i := 0; i < 3; i++ {
		_, ok := s.Next()
		if ok {
			t.Errorf("call %d after DONE returned ok=true", i)
		}
	}
}

func TestScanner_Err(t *testing.T) {
	input := "data: ok\n"
	s := NewScanner(strings.NewReader(input))
	s.Next()
	s.Next() // EOF

	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

func TestScanner_EmptyDataPayload(t *testing.T) {
	// "data:" with no space or value should yield an empty string token, not be skipped.
	input := "data:\n\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok {
		t.Fatal("expected ok=true for empty data payload, got false")
	}
	if data != "" {
		t.Errorf("expected empty string, got %q", data)
	}
}

// errReader is an io.Reader that returns a fixed error on every Read call.
type errReader struct{ err error }

func (e *errReader) Read(_ []byte) (int, error) { return 0, e.err }

func TestScanner_ReadError(t *testing.T) {
	injected := errors.New("stream broken")
	// First reader yields one valid event; second reader immediately returns the error.
	r := io.MultiReader(
		strings.NewReader("data: first\n\n"),
		&errReader{err: injected},
	)
	s := NewScanner(r)

	data, ok := s.Next()
	if !ok || data != "first" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "first")
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected ok=false after read error")
	}

	if err := s.Err(); !errors.Is(err, injected) {
		t.Errorf("Err() = %v; want injected error %v", err, injected)
	}
}

func TestScanner_MultiLineData(t *testing.T) {
	// Two consecutive data: lines in one event block.
	// The implementation does NOT concatenate; each data: line is returned independently.
	input := "data: line1\ndata: line2\n\n"
	s := NewScanner(strings.NewReader(input))

	data, ok := s.Next()
	if !ok || data != "line1" {
		t.Errorf("first: got %q, %v; want %q, true", data, ok, "line1")
	}

	data, ok = s.Next()
	if !ok || data != "line2" {
		t.Errorf("second: got %q, %v; want %q, true", data, ok, "line2")
	}

	_, ok = s.Next()
	if ok {
		t.Error("expected false after all lines consumed")
	}
}
